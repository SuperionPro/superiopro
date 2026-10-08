from dotenv import load_dotenv
from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from pydantic import BaseModel, Field, BeforeValidator
from typing import List, Optional, Annotated, Dict
from datetime import datetime, timezone, timedelta, date
from bson import ObjectId
import xml.etree.ElementTree as ET
import bcrypt
import jwt
import uuid
import json
import asyncio
from urllib.parse import quote_plus
import re
import io
import base64
import secrets
import hmac
import requests
import qrcode

# ------------------------------------------------------------------ DB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
import contextvars
CONTROL_DB_NAME = os.environ['DB_NAME']
cdb = client[CONTROL_DB_NAME]  # control DB: users, tenants, invites + primary tenant business data

_current_tenant_db = contextvars.ContextVar("tenant_db", default=None)


def tenant_db_for(tenant: dict):
    if not tenant or tenant.get("is_primary"):
        return cdb
    return client[f"{CONTROL_DB_NAME}_t_{str(tenant['_id'])}"]


class _TenantDBProxy:
    def __getattr__(self, name):
        tdb = _current_tenant_db.get()
        return (tdb if tdb is not None else cdb)[name]


db = _TenantDBProxy()

JWT_SECRET = os.environ['JWT_SECRET']
JWT_ALGORITHM = "HS256"
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

app = FastAPI(title="Superion Pro API")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superionpro")

# ------------------------------------------------------------------ Helpers
PyObjectId = Annotated[str, BeforeValidator(str)]

# Granular permission keys (match frontend labels)
ALL_PERMISSIONS = [
    "pos",            # Acesso ao PDV / Caixa
    "stock_entry",    # Entradas de Estoque / Importação de Nota Fiscal
    "costs",          # Visualizar Preço de Custo e Margem de Lucro
    "warehouse",      # Controle de Depósito / Almoxarifado
    "products",       # Cadastrar / Editar Produtos
    "service_orders", # Ordens de Serviço (OS)
    "commissions",    # Visualizar Relatório de Comissões
    "settings",       # Configurações do Sistema e Usuários
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7), "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def clean_user(u: dict) -> dict:
    u = dict(u)
    u["id"] = str(u.pop("_id"))
    u.pop("password_hash", None)
    u.pop("_tenant", None)
    return u


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else None
    if not token:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await cdb.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="Usuário não encontrado")
        tenant = None
        if user.get("tenant_id"):
            tenant = await cdb.tenants.find_one({"_id": ObjectId(user["tenant_id"])})
        if tenant:
            tenant = await resolve_tenant_status(tenant)
        else:
            tenant = {"is_primary": True, "status": "paid_active"}
        _current_tenant_db.set(tenant_db_for(tenant))
        user["_tenant"] = tenant
        user["_ip"] = _client_ip(request)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sessão expirada")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")


def has_permission(user: dict, perm: str) -> bool:
    if user.get("role") == "admin":
        return True
    return perm in (user.get("permissions") or [])


def require_permission(perm: str):
    async def checker(user: dict = Depends(get_current_user)):
        if not has_permission(user, perm):
            raise HTTPException(status_code=403, detail="Acesso Não Autorizado")
        return user
    return checker


# ------------------------------------------------------------------ Immutable Audit Log
AUDIT_PRODUCT_FIELDS = ["name", "ean", "code", "category", "cost", "price", "margin", "stock_store",
                        "stock_deposit", "min_stock", "unit", "expiry_date", "main_supplier_id",
                        "main_supplier_name"]


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def _snap(doc: dict, fields) -> dict:
    return {f: doc[f] for f in fields if f in doc}


def _diff(old: dict, new: dict):
    o, n = {}, {}
    for k in set(old) | set(new):
        if old.get(k) != new.get(k):
            o[k] = old.get(k)
            n[k] = new.get(k)
    return o, n


async def write_audit(user: dict, action_type: str, module: str, resource_id=None,
                      old_values=None, new_values=None, diff_only: bool = False, resource_name: str = ""):
    """Append an IMMUTABLE audit entry. No update/delete endpoints exist for audit_logs."""
    old_values = old_values or {}
    new_values = new_values or {}
    if diff_only:
        old_values, new_values = _diff(old_values, new_values)
        if not old_values and not new_values:
            return
    try:
        await db.audit_logs.insert_one({
            "timestamp": now_iso(),
            "user_id": str(user.get("_id", "")),
            "user_name": user.get("name", ""),
            "action_type": action_type,
            "module": module,
            "resource_id": str(resource_id) if resource_id else "",
            "resource_name": resource_name,
            "old_values": _json_safe(old_values),
            "new_values": _json_safe(new_values),
            "ip_address": user.get("_ip", ""),
        })
    except Exception as e:
        logger.warning(f"audit write failed: {e}")


class InventoryEditIn(BaseModel):
    name: str
    ean: str = ""
    code: str = ""
    category: str = ""
    cost: float = 0.0
    price: Optional[float] = None
    stock_store: float = 0.0
    expiry_date: str = ""
    main_supplier_id: str = ""
    main_supplier_name: str = ""
    unit: str = "Unidade"


# ------------------------------------------------------------------ Multi-tenant / subscription
TRIAL_DAYS = 10


async def resolve_tenant_status(tenant: dict) -> dict:
    if tenant.get("is_primary"):
        return tenant
    if tenant.get("status") == "in_trial":
        end = tenant.get("trial_end")
        expired = False
        try:
            if end:
                expired = datetime.now(timezone.utc) > datetime.fromisoformat(end)
        except Exception:
            expired = False
        if expired:
            await cdb.tenants.update_one({"_id": tenant["_id"]}, {"$set": {"status": "trial_expired"}})
            tenant["status"] = "trial_expired"
    return tenant


def is_superadmin(user: dict) -> bool:
    return bool((user.get("_tenant") or {}).get("is_primary")) and user.get("role") == "admin"


async def require_superadmin(user: dict = Depends(get_current_user)):
    if not is_superadmin(user):
        raise HTTPException(status_code=403, detail="Acesso restrito ao SuperAdmin")
    return user


async def require_active_tenant(user: dict = Depends(get_current_user)):
    tenant = user.get("_tenant") or {}
    if tenant.get("is_primary"):
        return user
    if tenant.get("status") == "trial_expired":
        raise HTTPException(status_code=402, detail={"paywall": True, "status": "trial_expired",
                            "message": "Seu período de teste expirou. Ative sua assinatura para continuar."})
    return user


def tenant_filter(user: dict, extra: dict = None):
    f = dict(extra or {})
    tid = user.get("tenant_id")
    if tid:
        f["tenant_id"] = tid
    return f


# ------------------------------------------------------------------ Models
class LoginInput(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    name: str
    username: str
    password: str
    role: str = "vendedor"
    permissions: List[str] = []


class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[List[str]] = None
    active: Optional[bool] = None


class PackagingUnit(BaseModel):
    name: str
    factor: float = 1.0


class ProductIn(BaseModel):
    name: str
    code: str = ""
    ean: str = ""
    brand: str = ""
    category: str = ""
    image: str = ""
    cost: float = 0.0
    cost_qty: float = 1.0
    margin: float = 30.0
    margin_wholesale: float = 15.0
    price: Optional[float] = None
    stock_store: float = 0.0
    stock_deposit: float = 0.0
    min_stock: float = 5.0
    unit: str = "Unidade"
    expiry_date: str = ""
    main_supplier_id: str = ""
    main_supplier_name: str = ""
    catalog_active: bool = True
    catalog_featured: bool = False
    catalog_price: Optional[float] = None
    catalog_image: str = ""
    catalog_description: str = ""
    packaging: List[PackagingUnit] = []
    # Fiscal
    ncm: str = ""
    cest: str = ""
    origem: str = "0"
    cfop: str = "5102"
    csosn: str = "102"
    pis_cst: str = "07"
    cofins_cst: str = "07"


class StockEntryIn(BaseModel):
    quantity: float
    unit_factor: float = 1.0
    cost_total: Optional[float] = None
    location: str = "deposit"  # store | deposit


class TransferIn(BaseModel):
    quantity: float
    direction: str = "deposit_to_store"  # deposit_to_store | store_to_deposit


class PaymentPart(BaseModel):
    method: str
    amount: float
    received: Optional[float] = None
    change: Optional[float] = None
    installments: Optional[int] = None
    card_flag: Optional[str] = None
    due_date: Optional[str] = None


class SaleItem(BaseModel):
    product_id: str
    name: str
    price: float
    cost: float = 0.0
    quantity: float


class SaleIn(BaseModel):
    items: List[SaleItem]
    client_name: str = "Consumidor Final"
    client_id: Optional[str] = None
    seller_id: str
    seller_name: str
    payments: List[PaymentPart]
    discount: float = 0.0


class SettingsIn(BaseModel):
    defaultTheme: Optional[str] = None
    defaultMargin: Optional[float] = None
    commissionRate: Optional[float] = None
    # Company / fiscal identity
    companyName: Optional[str] = None       # Razão Social
    tradeName: Optional[str] = None         # Nome Fantasia
    companyDoc: Optional[str] = None         # CNPJ
    ie: Optional[str] = None                 # Inscrição Estadual
    im: Optional[str] = None                 # Inscrição Municipal
    # Address
    addrStreet: Optional[str] = None
    addrNumber: Optional[str] = None
    addrNeighborhood: Optional[str] = None
    addrCity: Optional[str] = None
    addrState: Optional[str] = None
    addrZip: Optional[str] = None
    companyAddress: Optional[str] = None     # legacy single-line
    # Contact
    companyPhone: Optional[str] = None
    whatsapp: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    # Receipt / branding
    receiptFooter: Optional[str] = None
    logo: Optional[str] = None               # legacy
    logoLight: Optional[str] = None
    logoDark: Optional[str] = None
    logoReceipt: Optional[str] = None
    # Fiscal config (simulated SEFAZ)
    regimeTributario: Optional[str] = None   # simples | presumido | real
    fiscalEnvironment: Optional[str] = None  # homologacao | producao
    cscId: Optional[str] = None
    cscToken: Optional[str] = None
    certFilename: Optional[str] = None
    certUploaded: Optional[bool] = None
    # Appearance / custom palette
    colorPrimary: Optional[str] = None
    colorSecondary: Optional[str] = None
    colorBackground: Optional[str] = None
    colorText: Optional[str] = None


class InvoiceItemIn(BaseModel):
    ean: str = ""
    code: str = ""
    name: str
    quantity: float
    unit_cost: float
    matched_product_id: Optional[str] = None
    create_new: bool = False
    margin: float = 30.0


class InvoiceImportIn(BaseModel):
    supplier: str = ""
    number: str = ""
    issue_date: str = ""
    location: str = "deposit"  # store | deposit
    items: List[InvoiceItemIn]


class AssistantIn(BaseModel):
    message: str


# ------------------------------------------------------------------ Auth
@api_router.post("/auth/login")
async def login(data: LoginInput):
    user = await cdb.users.find_one({"username": data.username.lower().strip()})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos")
    if user.get("active") is False:
        raise HTTPException(status_code=403, detail="Usuário desativado")
    token = create_token(str(user["_id"]))
    return {"token": token, "user": clean_user(user)}


@api_router.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return clean_user(user)


# ------------------------------------------------------------------ Users
@api_router.get("/users")
async def list_users(user: dict = Depends(require_permission("settings"))):
    users = await cdb.users.find(tenant_filter(user)).sort("created_at", -1).to_list(500)
    return [clean_user(u) for u in users]


@api_router.get("/users/sellers")
async def list_sellers(user: dict = Depends(get_current_user)):
    users = await cdb.users.find(tenant_filter(user, {"active": {"$ne": False}})).to_list(500)
    return [{"id": str(u["_id"]), "name": u["name"], "role": u["role"]} for u in users]


@api_router.post("/users")
async def create_user(data: UserCreate, user: dict = Depends(require_permission("settings"))):
    uname = data.username.lower().strip()
    if await cdb.users.find_one({"username": uname}):
        raise HTTPException(status_code=400, detail="Nome de usuário já existe")
    perms = [p for p in data.permissions if p in ALL_PERMISSIONS]
    doc = {"name": data.name, "username": uname, "password_hash": hash_password(data.password),
           "role": data.role, "permissions": perms, "active": True, "created_at": now_iso(),
           "tenant_id": user.get("tenant_id")}
    res = await cdb.users.insert_one(doc)
    doc["_id"] = res.inserted_id
    return clean_user(doc)


@api_router.put("/users/{user_id}")
async def update_user(user_id: str, data: UserUpdate, user: dict = Depends(require_permission("settings"))):
    update = {}
    if data.name is not None:
        update["name"] = data.name
    if data.password:
        update["password_hash"] = hash_password(data.password)
    if data.role is not None:
        update["role"] = data.role
    if data.permissions is not None:
        update["permissions"] = [p for p in data.permissions if p in ALL_PERMISSIONS]
    if data.active is not None:
        update["active"] = data.active
    if update:
        await cdb.users.update_one(tenant_filter(user, {"_id": ObjectId(user_id)}), {"$set": update})
    doc = await cdb.users.find_one(tenant_filter(user, {"_id": ObjectId(user_id)}))
    return clean_user(doc)


@api_router.delete("/users/{user_id}")
async def delete_user(user_id: str, user: dict = Depends(require_permission("settings"))):
    target = await cdb.users.find_one(tenant_filter(user, {"_id": ObjectId(user_id)}))
    if target and target.get("role") == "admin":
        raise HTTPException(status_code=400, detail="Não é possível excluir o administrador principal")
    await cdb.users.delete_one(tenant_filter(user, {"_id": ObjectId(user_id)}))
    return {"ok": True}


# ------------------------------------------------------------------ Products
def compute_price(cost: float, cost_qty: float, margin: float, override: Optional[float]) -> float:
    if override is not None and override > 0:
        return round(override, 2)
    base = (cost / cost_qty) if cost_qty else 0.0
    return round(base * (1 + margin / 100.0), 2)


def product_out(p: dict) -> dict:
    p = dict(p)
    p["id"] = str(p.pop("_id"))
    p["stock"] = round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 3)
    return p


@api_router.get("/products")
async def list_products(user: dict = Depends(get_current_user)):
    products = await db.products.find().sort("name", 1).to_list(2000)
    out = [product_out(p) for p in products]
    if not has_permission(user, "costs"):
        for p in out:
            p.pop("cost", None)
            p.pop("margin", None)
            p.pop("cost_qty", None)
    return out


@api_router.get("/products/metrics")
async def product_metrics(user: dict = Depends(get_current_user)):
    products = await db.products.find().to_list(2000)
    def total_stock(p):
        return p.get("stock_store", 0) + p.get("stock_deposit", 0)
    total = len(products)
    active = sum(1 for p in products if p.get("active", True))
    low = sum(1 for p in products if 0 < total_stock(p) <= p.get("min_stock", 5))
    out = sum(1 for p in products if total_stock(p) <= 0)
    return {"total": total, "active": active, "low_stock": low, "out_of_stock": out}


@api_router.post("/products")
async def create_product(data: ProductIn, user: dict = Depends(require_permission("products"))):
    doc = data.model_dump()
    doc["price"] = compute_price(data.cost, data.cost_qty, data.margin, data.price)
    doc["price_wholesale"] = compute_price(data.cost, data.cost_qty, data.margin_wholesale, None)
    doc["active"] = True
    doc["created_at"] = now_iso()
    res = await db.products.insert_one(doc)
    doc["_id"] = res.inserted_id
    await write_audit(user, "CREATE", "Produtos", res.inserted_id,
                      new_values=_snap(doc, AUDIT_PRODUCT_FIELDS), resource_name=doc.get("name", ""))
    return product_out(doc)


@api_router.put("/products/{pid}")
async def update_product(pid: str, data: ProductIn, user: dict = Depends(require_permission("products"))):
    old = await db.products.find_one({"_id": ObjectId(pid)}) or {}
    doc = data.model_dump()
    doc["price"] = compute_price(data.cost, data.cost_qty, data.margin, data.price)
    doc["price_wholesale"] = compute_price(data.cost, data.cost_qty, data.margin_wholesale, None)
    await db.products.update_one({"_id": ObjectId(pid)}, {"$set": doc})
    p = await db.products.find_one({"_id": ObjectId(pid)})
    await write_audit(user, "UPDATE", "Produtos", pid, _snap(old, AUDIT_PRODUCT_FIELDS),
                      _snap(p, AUDIT_PRODUCT_FIELDS), diff_only=True, resource_name=p.get("name", ""))
    return product_out(p)


@api_router.post("/products/{pid}/stock")
async def stock_entry(pid: str, data: StockEntryIn, user: dict = Depends(require_permission("stock_entry"))):
    p = await db.products.find_one({"_id": ObjectId(pid)})
    if not p:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    old_store, old_deposit = p.get("stock_store", 0), p.get("stock_deposit", 0)
    base_units = data.quantity * data.unit_factor
    field = "stock_store" if data.location == "store" else "stock_deposit"
    update = {field: p.get(field, 0) + base_units}
    if data.cost_total and base_units:
        new_cost = round(data.cost_total, 2)
        update["cost"] = new_cost
        update["cost_qty"] = base_units
        update["price"] = compute_price(new_cost, base_units, p.get("margin", 30), None)
    await db.products.update_one({"_id": ObjectId(pid)}, {"$set": update})
    await db.stock_entries.insert_one({"product_id": pid, "quantity": data.quantity, "unit_factor": data.unit_factor,
                                       "base_units": base_units, "cost_total": data.cost_total, "location": data.location,
                                       "created_at": now_iso(), "user": user["name"]})
    p = await db.products.find_one({"_id": ObjectId(pid)})
    await write_audit(user, "UPDATE", "Estoque", pid,
                      old_values={"stock_store": old_store, "stock_deposit": old_deposit},
                      new_values={"stock_store": p.get("stock_store", 0), "stock_deposit": p.get("stock_deposit", 0),
                                  "entrada_qtd": base_units, "local": data.location}, resource_name=p.get("name", ""))
    return product_out(p)


@api_router.post("/products/{pid}/transfer")
async def transfer_stock(pid: str, data: TransferIn, user: dict = Depends(require_permission("warehouse"))):
    p = await db.products.find_one({"_id": ObjectId(pid)})
    if not p:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    store = p.get("stock_store", 0)
    deposit = p.get("stock_deposit", 0)
    old_store, old_deposit = store, deposit
    if data.direction == "deposit_to_store":
        if data.quantity > deposit:
            raise HTTPException(status_code=400, detail="Quantidade maior que o estoque do depósito")
        store += data.quantity
        deposit -= data.quantity
    else:
        if data.quantity > store:
            raise HTTPException(status_code=400, detail="Quantidade maior que o estoque da loja")
        store -= data.quantity
        deposit += data.quantity
    await db.products.update_one({"_id": ObjectId(pid)}, {"$set": {"stock_store": store, "stock_deposit": deposit}})
    p = await db.products.find_one({"_id": ObjectId(pid)})
    await write_audit(user, "UPDATE", "Estoque", pid,
                      old_values={"stock_store": old_store, "stock_deposit": old_deposit},
                      new_values={"stock_store": store, "stock_deposit": deposit, "transferencia": data.direction},
                      resource_name=p.get("name", ""))
    return product_out(p)


@api_router.delete("/products/{pid}")
async def delete_product(pid: str, user: dict = Depends(require_permission("products"))):
    p = await db.products.find_one({"_id": ObjectId(pid)})
    if p:
        await db.deleted_products.insert_one({
            "product_id": pid, "name": p.get("name", ""), "ean": p.get("ean", ""),
            "code": p.get("code", ""), "category": p.get("category", ""),
            "stock": round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 3),
            "cost": p.get("cost", 0), "price": p.get("price", 0),
            "deleted_by": user["name"], "deleted_at": now_iso()})
    await db.products.delete_one({"_id": ObjectId(pid)})
    await write_audit(user, "DELETE", "Produtos", pid,
                      old_values=_snap(p or {}, AUDIT_PRODUCT_FIELDS), resource_name=(p or {}).get("name", ""))
    return {"ok": True}


@api_router.put("/v1/inventory/{pid}")
async def inventory_edit(pid: str, data: InventoryEditIn, user: dict = Depends(require_permission("products"))):
    old = await db.products.find_one({"_id": ObjectId(pid)})
    if not old:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    upd = data.model_dump()
    upd["price"] = compute_price(data.cost, old.get("cost_qty", 1.0), old.get("margin", 30.0), data.price)
    await db.products.update_one({"_id": ObjectId(pid)}, {"$set": upd})
    new = await db.products.find_one({"_id": ObjectId(pid)})
    await write_audit(user, "UPDATE", "Inventário", pid, _snap(old, AUDIT_PRODUCT_FIELDS),
                      _snap(new, AUDIT_PRODUCT_FIELDS), diff_only=True, resource_name=new.get("name", ""))
    return product_out(new)


@api_router.get("/audit-logs")
async def list_audit_logs(date_from: str = "", date_to: str = "", user_id: str = "", action_type: str = "",
                          module: str = "", user: dict = Depends(require_permission("settings"))):
    q = {}
    if user_id:
        q["user_id"] = user_id
    if action_type:
        q["action_type"] = action_type
    if module:
        q["module"] = module
    if date_from or date_to:
        tr = {}
        if date_from:
            tr["$gte"] = date_from
        if date_to:
            tr["$lte"] = date_to + "T23:59:59"
        q["timestamp"] = tr
    logs = await db.audit_logs.find(q).sort("timestamp", -1).to_list(2000)
    for l in logs:
        l["id"] = str(l.pop("_id"))
    return logs


@api_router.get("/audit-logs/filters")
async def audit_filters(user: dict = Depends(require_permission("settings"))):
    users = await db.audit_logs.distinct("user_name")
    modules = await db.audit_logs.distinct("module")
    return {"users": sorted([u for u in users if u]), "modules": sorted([m for m in modules if m])}


# ------------------------------------------------------------------ Barcode lookup (web)
@api_router.get("/products/lookup/{ean}")
async def barcode_lookup(ean: str, user: dict = Depends(get_current_user)):
    ean = ean.strip()
    if not ean:
        raise HTTPException(status_code=400, detail="Código de barras vazio")
    # already in local inventory?
    local = await db.products.find_one({"ean": ean})
    if local:
        return {"found": True, "source": "local", "name": local["name"], "brand": "",
                "quantity": local.get("unit", ""), "description": "", "category": local.get("category", ""),
                "image": local.get("image", ""), "ncm": local.get("ncm", ""), "existing_id": str(local["_id"])}
    # Open Food Facts (free, no key)
    try:
        r = requests.get(f"https://world.openfoodfacts.org/api/v0/product/{ean}.json", timeout=8,
                         headers={"User-Agent": "SuperionPro/1.0"})
        data = r.json()
        if data.get("status") == 1:
            p = data["product"]
            name = p.get("product_name_pt") or p.get("product_name") or p.get("generic_name") or ""
            brand = (p.get("brands") or "").split(",")[0].strip()
            image = p.get("image_front_url") or p.get("image_url") or ""
            cats = p.get("categories") or ""
            category = cats.split(",")[-1].strip() if cats else ""
            if name:
                return {"found": True, "source": "openfoodfacts", "name": name,
                        "brand": brand, "quantity": p.get("quantity", ""), "description": p.get("generic_name", ""),
                        "category": category, "image": image, "ncm": ""}
    except Exception as e:
        logger.warning(f"openfoodfacts lookup failed: {e}")
    # UPCitemdb (free trial, no key) — broader general-product coverage
    try:
        r = requests.get(f"https://api.upcitemdb.com/prod/trial/lookup?upc={ean}", timeout=8,
                         headers={"User-Agent": "SuperionPro/1.0"})
        data = r.json()
        items = data.get("items") or []
        if items:
            it = items[0]
            name = (it.get("title") or "").strip()
            brand = (it.get("brand") or "").strip()
            category = (it.get("category") or "").split(">")[-1].strip()
            imgs = it.get("images") or []
            image = imgs[0] if imgs else ""
            if name:
                return {"found": True, "source": "upcitemdb", "name": name,
                        "brand": brand, "quantity": "", "description": it.get("description", ""),
                        "category": category, "image": image, "ncm": ""}
    except Exception as e:
        logger.warning(f"upcitemdb lookup failed: {e}")
    return {"found": False, "source": "none", "name": "", "brand": "", "quantity": "", "description": "",
            "category": "", "image": "", "ncm": ""}


# ------------------------------------------------------------------ Lots & expiration
class LotIn(BaseModel):
    lot_number: str
    expiry: str  # YYYY-MM-DD
    quantity: float
    location: str = "deposit"  # store | deposit


def _days_to_expiry(expiry: str):
    try:
        return (date.fromisoformat(expiry[:10]) - date.today()).days
    except Exception:
        return None


def lot_out(l: dict) -> dict:
    l = dict(l)
    l["id"] = str(l.pop("_id"))
    l["days_to_expiry"] = _days_to_expiry(l.get("expiry", ""))
    return l


@api_router.post("/products/{pid}/lots")
async def add_lot(pid: str, data: LotIn, user: dict = Depends(require_permission("stock_entry"))):
    p = await db.products.find_one({"_id": ObjectId(pid)})
    if not p:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    if data.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantidade do lote deve ser maior que zero")
    if _days_to_expiry(data.expiry) is None:
        raise HTTPException(status_code=400, detail="Data de validade inválida (use AAAA-MM-DD)")
    field = "stock_store" if data.location == "store" else "stock_deposit"
    await db.products.update_one({"_id": ObjectId(pid)}, {"$inc": {field: data.quantity}})
    doc = {"product_id": pid, "product_name": p["name"], "lot_number": data.lot_number,
           "expiry": data.expiry[:10], "quantity": data.quantity, "location": data.location,
           "created_at": now_iso()}
    res = await db.lots.insert_one(doc)
    doc["_id"] = res.inserted_id
    return lot_out(doc)


@api_router.get("/products/{pid}/lots")
async def list_product_lots(pid: str, user: dict = Depends(get_current_user)):
    lots = await db.lots.find({"product_id": pid}).sort("expiry", 1).to_list(500)
    return [lot_out(l) for l in lots]


@api_router.delete("/lots/{lot_id}")
async def delete_lot(lot_id: str, user: dict = Depends(require_permission("stock_entry"))):
    lot = await db.lots.find_one({"_id": ObjectId(lot_id)})
    if lot:
        field = "stock_store" if lot.get("location") == "store" else "stock_deposit"
        await db.products.update_one({"_id": ObjectId(lot["product_id"])}, {"$inc": {field: -lot.get("quantity", 0)}})
        await db.lots.delete_one({"_id": ObjectId(lot_id)})
    return {"ok": True}


@api_router.get("/inventory")
async def inventory(q: str = "", user: dict = Depends(get_current_user),
                    _gate: dict = Depends(require_active_tenant)):
    products = await db.products.find().sort("name", 1).to_list(2000)
    all_lots = await db.lots.find().sort("expiry", 1).to_list(5000)
    lots_by_pid: Dict[str, list] = {}
    for l in all_lots:
        lots_by_pid.setdefault(l["product_id"], []).append(lot_out(l))
    ql = q.lower().strip()
    out = []
    for p in products:
        if ql and not (ql in p["name"].lower() or ql in (p.get("ean", "") or "") or ql in (p.get("code", "") or "").lower() or ql in (p.get("category", "") or "").lower()):
            continue
        out.append({
            "id": str(p["_id"]), "name": p["name"], "ean": p.get("ean", ""), "code": p.get("code", ""),
            "category": p.get("category", ""), "image": p.get("image", ""), "price": p.get("price", 0),
            "cost": p.get("cost", 0), "unit": p.get("unit", "Unidade"),
            "expiry_date": p.get("expiry_date", ""), "main_supplier_id": p.get("main_supplier_id", ""),
            "main_supplier_name": p.get("main_supplier_name", ""),
            "stock_store": p.get("stock_store", 0), "stock_deposit": p.get("stock_deposit", 0),
            "lots": lots_by_pid.get(str(p["_id"]), []),
        })
    return out


@api_router.get("/inventory/expiring")
async def inventory_expiring(days: int = 15, user: dict = Depends(get_current_user),
                            _gate: dict = Depends(require_active_tenant)):
    """Produtos e lotes vencendo nos próximos N dias (default 15), para montar promoções."""
    prods = {str(p["_id"]): p for p in await db.products.find().to_list(5000)}
    alerts = []
    # product-level expiry
    for pid, p in prods.items():
        d = _days_to_expiry(p.get("expiry_date", "")) if p.get("expiry_date") else None
        if d is not None and d <= days:
            stock = round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 3)
            price = p.get("price", 0)
            alerts.append({"product_id": pid, "name": p.get("name"), "ean": p.get("ean", ""),
                           "category": p.get("category", ""), "image": p.get("image", ""),
                           "source": "produto", "lot_number": "", "quantity": stock,
                           "stock_store": p.get("stock_store", 0), "stock_deposit": p.get("stock_deposit", 0),
                           "price": price, "cost": p.get("cost", 0), "expiry_date": p.get("expiry_date", "")[:10],
                           "days_to_expiry": d, "promo_price": round(price * 0.75, 2)})
    # lot-level expiry
    for l in await db.lots.find().to_list(10000):
        d = _days_to_expiry(l.get("expiry", ""))
        if d is not None and d <= days:
            p = prods.get(l.get("product_id"), {})
            price = p.get("price", 0)
            alerts.append({"product_id": l.get("product_id"), "name": p.get("name", "Produto"),
                           "ean": p.get("ean", ""), "category": p.get("category", ""), "image": p.get("image", ""),
                           "source": "lote", "lot_number": l.get("lot_number", ""), "quantity": l.get("quantity", 0),
                           "stock_store": p.get("stock_store", 0), "stock_deposit": p.get("stock_deposit", 0),
                           "price": price, "cost": p.get("cost", 0), "expiry_date": l.get("expiry", "")[:10],
                           "days_to_expiry": d, "promo_price": round(price * 0.75, 2)})
    alerts.sort(key=lambda a: a["days_to_expiry"])
    return {"days": days, "count": len(alerts),
            "expired": sum(1 for a in alerts if a["days_to_expiry"] < 0),
            "items": alerts}


# ------------------------------------------------------------------ Invoice (NFe) import
NFE_NS = {"n": "http://www.portalfiscal.inf.br/nfe"}


def _find(el, path):
    r = el.find(path, NFE_NS)
    return r.text if r is not None and r.text else ""


def parse_nfe_xml(content: bytes) -> dict:
    text = content.decode("utf-8", errors="ignore")
    text = re.sub(r'xmlns="[^"]+"', '', text, count=0)  # keep, handled by ns
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"XML inválido: {e}")
    # locate infNFe regardless of nesting
    infnfe = root.find(".//n:infNFe", NFE_NS)
    if infnfe is None:
        raise HTTPException(status_code=400, detail="Não é um XML de NFe válido")
    emit = infnfe.find("n:emit", NFE_NS)
    ide = infnfe.find("n:ide", NFE_NS)
    supplier = _find(emit, "n:xNome") if emit is not None else ""
    number = _find(ide, "n:nNF") if ide is not None else ""
    issue = (_find(ide, "n:dhEmi") or _find(ide, "n:dEmi")) if ide is not None else ""
    items = []
    for det in infnfe.findall("n:det", NFE_NS):
        prod = det.find("n:prod", NFE_NS)
        if prod is None:
            continue
        ean = _find(prod, "n:cEAN")
        if ean.upper() in ("SEM GTIN", ""):
            ean = ""
        try:
            qty = float(_find(prod, "n:qCom") or 0)
        except ValueError:
            qty = 0
        try:
            unit_cost = float(_find(prod, "n:vUnCom") or 0)
        except ValueError:
            unit_cost = 0
        items.append({"code": _find(prod, "n:cProd"), "ean": ean, "name": _find(prod, "n:xProd"),
                      "quantity": round(qty, 3), "unit_cost": round(unit_cost, 4)})
    return {"supplier": supplier, "number": number, "issue_date": issue[:10], "items": items}


@api_router.post("/invoices/parse")
async def parse_invoice(file: UploadFile = File(...), user: dict = Depends(require_permission("stock_entry"))):
    content = await file.read()
    fname = (file.filename or "").lower()
    if fname.endswith(".pdf") or (not fname.endswith(".xml") and content[:4] == b"%PDF"):
        raise HTTPException(status_code=400, detail="Leitura de PDF indisponível no momento. Envie o arquivo XML da NFe (NFe/NFCe) para importação automática.")
    parsed = parse_nfe_xml(content)
    # match products
    for it in parsed["items"]:
        match = None
        if it["ean"]:
            match = await db.products.find_one({"ean": it["ean"]})
        if not match and it["code"]:
            match = await db.products.find_one({"code": it["code"]})
        it["matched_product_id"] = str(match["_id"]) if match else None
        it["matched_name"] = match["name"] if match else None
    return parsed


@api_router.post("/invoices/import")
async def import_invoice(data: InvoiceImportIn, user: dict = Depends(require_permission("stock_entry"))):
    field = "stock_store" if data.location == "store" else "stock_deposit"
    created, updated = 0, 0
    for it in data.items:
        if it.matched_product_id:
            p = await db.products.find_one({"_id": ObjectId(it.matched_product_id)})
            if not p:
                continue
            new_stock = p.get(field, 0) + it.quantity
            new_price = compute_price(it.unit_cost, 1.0, p.get("margin", 30), None)
            await db.products.update_one({"_id": p["_id"]}, {"$set": {field: new_stock, "cost": it.unit_cost,
                                                                      "cost_qty": 1.0, "price": new_price}})
            updated += 1
        elif it.create_new:
            price = compute_price(it.unit_cost, 1.0, it.margin, None)
            doc = {"name": it.name, "code": it.code, "ean": it.ean, "category": "", "image": "",
                   "cost": it.unit_cost, "cost_qty": 1.0, "margin": it.margin, "price": price,
                   "stock_store": it.quantity if data.location == "store" else 0,
                   "stock_deposit": it.quantity if data.location == "deposit" else 0,
                   "min_stock": 5.0, "unit": "Unidade", "packaging": [], "active": True, "created_at": now_iso()}
            await db.products.insert_one(doc)
            created += 1
    await db.invoices.insert_one({"supplier": data.supplier, "number": data.number, "issue_date": data.issue_date,
                                  "location": data.location, "item_count": len(data.items), "created": created,
                                  "updated": updated, "created_at": now_iso(), "user": user["name"]})
    return {"ok": True, "created": created, "updated": updated}


@api_router.get("/invoices")
async def list_invoices(user: dict = Depends(require_permission("stock_entry"))):
    invs = await db.invoices.find().sort("created_at", -1).to_list(200)
    for i in invs:
        i["id"] = str(i.pop("_id"))
    return invs


# ------------------------------------------------------------------ Sales (POS)
@api_router.post("/sales")
async def create_sale(data: SaleIn, user: dict = Depends(require_permission("pos")),
                      _gate: dict = Depends(require_active_tenant)):
    subtotal = sum(i.price * i.quantity for i in data.items)
    total = round(subtotal - data.discount, 2)
    paid = round(sum(p.amount for p in data.payments), 2)
    if abs(paid - total) > 0.01:
        raise HTTPException(status_code=400, detail=f"Pagamento (R$ {paid:.2f}) difere do total (R$ {total:.2f})")
    cost_total = sum(i.cost * i.quantity for i in data.items)
    settings = await db.settings.find_one({"_id": "general"}) or {}
    rate = settings.get("commissionRate", 5.0)
    commission = round(total * rate / 100.0, 2)
    doc = data.model_dump()
    doc.update({"subtotal": round(subtotal, 2), "total": total, "cost_total": round(cost_total, 2),
                "profit": round(total - cost_total, 2), "commission": commission, "commission_rate": rate,
                "commission_paid": False, "cashier_id": str(user["_id"]), "cashier_name": user["name"],
                "source": "pos", "receipt_no": str(uuid.uuid4().int)[:8], "created_at": now_iso()})
    res = await db.sales.insert_one(doc)
    for it in data.items:
        await db.products.update_one({"_id": ObjectId(it.product_id)}, {"$inc": {"stock_store": -it.quantity}})
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    await write_audit(user, "CREATE", "PDV", doc["id"], new_values={
        "receipt_no": doc["receipt_no"], "total": doc["total"], "cliente": doc.get("client_name"),
        "itens": len(doc.get("items", [])), "pagamentos": [p.get("method") for p in doc.get("payments", [])]},
        resource_name=f"Venda #{doc['receipt_no']}")
    return doc


@api_router.get("/sales")
async def list_sales(user: dict = Depends(get_current_user)):
    sales = await db.sales.find().sort("created_at", -1).to_list(1000)
    for s in sales:
        s["id"] = str(s.pop("_id"))
    return sales


async def _restore_sale_stock(items):
    for it in items or []:
        pid = it.get("product_id")
        if pid and ObjectId.is_valid(pid):
            await db.products.update_one({"_id": ObjectId(pid)}, {"$inc": {"stock_store": it.get("quantity", 0)}})


@api_router.put("/sales/{sale_id}")
async def update_sale(sale_id: str, data: SaleIn, user: dict = Depends(require_permission("pos")),
                      _gate: dict = Depends(require_active_tenant)):
    existing = await db.sales.find_one({"_id": ObjectId(sale_id)})
    if not existing:
        raise HTTPException(status_code=404, detail="Venda não encontrada")
    if existing.get("fiscal", {}).get("status") == "autorizada":
        raise HTTPException(status_code=400, detail="Venda com NFC-e autorizada não pode ser editada")
    subtotal = sum(i.price * i.quantity for i in data.items)
    total = round(subtotal - data.discount, 2)
    paid = round(sum(p.amount for p in data.payments), 2)
    if abs(paid - total) > 0.01:
        raise HTTPException(status_code=400, detail=f"Pagamento (R$ {paid:.2f}) difere do total (R$ {total:.2f})")
    # restore previous stock, then deduct the new quantities
    await _restore_sale_stock(existing.get("items", []))
    for it in data.items:
        if ObjectId.is_valid(it.product_id):
            await db.products.update_one({"_id": ObjectId(it.product_id)}, {"$inc": {"stock_store": -it.quantity}})
    cost_total = sum(i.cost * i.quantity for i in data.items)
    rate = existing.get("commission_rate", 5.0)
    upd = data.model_dump()
    upd.update({"subtotal": round(subtotal, 2), "total": total, "cost_total": round(cost_total, 2),
                "profit": round(total - cost_total, 2), "commission": round(total * rate / 100.0, 2),
                "commission_rate": rate, "updated_at": now_iso()})
    await db.sales.update_one({"_id": ObjectId(sale_id)}, {"$set": upd})
    doc = await db.sales.find_one({"_id": ObjectId(sale_id)})
    doc["id"] = str(doc.pop("_id"))
    await write_audit(user, "UPDATE", "PDV", sale_id,
                      old_values={"total": existing.get("total"), "itens": len(existing.get("items", []))},
                      new_values={"total": doc.get("total"), "itens": len(doc.get("items", []))},
                      resource_name=f"Venda #{doc.get('receipt_no')}")
    return doc


@api_router.delete("/sales/{sale_id}")
async def delete_sale(sale_id: str, user: dict = Depends(require_permission("pos"))):
    existing = await db.sales.find_one({"_id": ObjectId(sale_id)})
    if not existing:
        raise HTTPException(status_code=404, detail="Venda não encontrada")
    if existing.get("fiscal", {}).get("status") == "autorizada":
        raise HTTPException(status_code=400, detail="Venda com NFC-e autorizada não pode ser cancelada")
    await _restore_sale_stock(existing.get("items", []))
    await db.sales.delete_one({"_id": ObjectId(sale_id)})
    await write_audit(user, "DELETE", "PDV", sale_id,
                      old_values={"receipt_no": existing.get("receipt_no"), "total": existing.get("total"),
                                  "cliente": existing.get("client_name")},
                      resource_name=f"Venda #{existing.get('receipt_no')}")
    return {"ok": True}


# ------------------------------------------------------------------ Fiscal (SIMULATED NFC-e / SEFAZ)
def _qr_data_url(payload: str) -> str:
    img = qrcode.make(payload)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _gen_access_key() -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(44))


def _tax_engine(sale: dict, regime: str) -> dict:
    total = sale.get("total", 0)
    # Simplified simulated tax summary
    if regime == "simples":
        icms = 0.0
    else:
        icms = round(total * 0.18, 2)
    pis = round(total * 0.0165, 2)
    cofins = round(total * 0.076, 2)
    return {"icms": icms, "pis": pis, "cofins": cofins, "regime": regime}


@api_router.post("/fiscal/certificate")
async def upload_certificate(file: UploadFile = File(...), user: dict = Depends(require_permission("settings"))):
    fname = file.filename or "certificado.pfx"
    await file.read()  # not persisted (simulated)
    await db.settings.update_one({"_id": "general"}, {"$set": {"certFilename": fname, "certUploaded": True}}, upsert=True)
    return {"ok": True, "certFilename": fname}


class EmitNfceIn(BaseModel):
    cpf_cnpj: Optional[str] = None


@api_router.post("/sales/{sale_id}/emit-nfce")
async def emit_nfce(sale_id: str, data: Optional[EmitNfceIn] = None, user: dict = Depends(require_permission("pos"))):
    sale = await db.sales.find_one({"_id": ObjectId(sale_id)})
    if not sale:
        raise HTTPException(status_code=404, detail="Venda não encontrada")
    if sale.get("fiscal", {}).get("status") == "autorizada":
        sale["id"] = str(sale.pop("_id"))
        return sale["fiscal"]
    settings = await db.settings.find_one({"_id": "general"}) or {}
    if not settings.get("certUploaded"):
        raise HTTPException(status_code=400, detail="Certificado digital A1 não configurado. Acesse Módulo Fiscal.")
    env = settings.get("fiscalEnvironment", "homologacao")
    regime = settings.get("regimeTributario", "simples")
    access_key = _gen_access_key()
    protocol = "135" + "".join(str(secrets.randbelow(10)) for _ in range(12))
    authorized_at = now_iso()
    qr_payload = (f"https://www.{'homologacao.' if env=='homologacao' else ''}nfce.sefaz/consulta?"
                  f"p={access_key}|2|{'2' if env=='homologacao' else '1'}|1|{settings.get('cscId','000001')}")
    taxes = _tax_engine(sale, regime)
    xml_sample = (f"<?xml version='1.0' encoding='UTF-8'?>\n"
                  f"<!-- XML NFC-e SIMULADO (MOCK) - nao valido fiscalmente -->\n"
                  f"<nfeProc versao='4.00'>\n  <NFe>\n    <infNFe Id='NFe{access_key}'>\n"
                  f"      <ide><mod>65</mod><tpAmb>{'2' if env=='homologacao' else '1'}</tpAmb></ide>\n"
                  f"      <emit><xNome>{settings.get('companyName','')}</xNome>"
                  f"<CNPJ>{settings.get('companyDoc','')}</CNPJ></emit>\n"
                  f"      <total><vNF>{sale.get('total',0):.2f}</vNF>"
                  f"<vICMS>{taxes['icms']:.2f}</vICMS></total>\n    </infNFe>\n  </NFe>\n"
                  f"  <protNFe><infProt><nProt>{protocol}</nProt>"
                  f"<cStat>100</cStat><xMotivo>Autorizado o uso da NF-e</xMotivo></infProt></protNFe>\n</nfeProc>")
    fiscal = {
        "status": "autorizada", "mock": True, "model": "NFC-e (65)", "environment": env,
        "access_key": access_key, "protocol": protocol, "authorized_at": authorized_at,
        "qr_payload": qr_payload, "qr_image": _qr_data_url(qr_payload), "taxes": taxes,
        "xml": xml_sample, "series": "1", "number": sale.get("receipt_no"),
        "customer_doc": (data.cpf_cnpj if data else None),
    }
    await db.sales.update_one({"_id": ObjectId(sale_id)}, {"$set": {"fiscal": fiscal}})
    return fiscal


@api_router.get("/fiscal/logs")
async def fiscal_logs(user: dict = Depends(require_permission("settings"))):
    sales = await db.sales.find({"fiscal": {"$exists": True}}).sort("created_at", -1).to_list(500)
    return [{"id": str(s["_id"]), "receipt_no": s.get("receipt_no"), "total": s.get("total"),
             "status": s.get("fiscal", {}).get("status"), "access_key": s.get("fiscal", {}).get("access_key"),
             "protocol": s.get("fiscal", {}).get("protocol"), "environment": s.get("fiscal", {}).get("environment"),
             "created_at": s.get("created_at")} for s in sales]


# ------------------------------------------------------------------ Commissions
@api_router.get("/commissions")
async def commissions(user: dict = Depends(require_permission("commissions"))):
    sales = await db.sales.find().to_list(5000)
    total_sales = round(sum(s.get("total", 0) for s in sales), 2)
    total_comm = round(sum(s.get("commission", 0) for s in sales), 2)
    paid = round(sum(s.get("commission", 0) for s in sales if s.get("commission_paid")), 2)
    pending = round(total_comm - paid, 2)
    ranking: Dict[str, dict] = {}
    for s in sales:
        key = s.get("seller_name", "—")
        r = ranking.setdefault(key, {"seller": key, "sales": 0.0, "commission": 0.0, "count": 0})
        r["sales"] += s.get("total", 0)
        r["commission"] += s.get("commission", 0)
        r["count"] += 1
    for r in ranking.values():
        r["sales"] = round(r["sales"], 2)
        r["commission"] = round(r["commission"], 2)
    items = [{"id": str(s["_id"]), "receipt_no": s.get("receipt_no"), "seller_name": s.get("seller_name"),
              "total": s.get("total"), "commission": s.get("commission"), "paid": s.get("commission_paid", False),
              "source": s.get("source", "pos"), "created_at": s.get("created_at")} for s in sales]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"metrics": {"total_sales": total_sales, "total_commission": total_comm, "paid": paid, "pending": pending},
            "ranking": sorted(ranking.values(), key=lambda x: x["commission"], reverse=True), "items": items}


@api_router.post("/commissions/{sale_id}/toggle")
async def toggle_commission(sale_id: str, user: dict = Depends(require_permission("commissions"))):
    s = await db.sales.find_one({"_id": ObjectId(sale_id)})
    if not s:
        raise HTTPException(status_code=404, detail="Venda não encontrada")
    new_val = not s.get("commission_paid", False)
    await db.sales.update_one({"_id": ObjectId(sale_id)}, {"$set": {"commission_paid": new_val}})
    return {"ok": True, "paid": new_val}


# ------------------------------------------------------------------ Dashboard
@api_router.get("/dashboard")
async def dashboard(user: dict = Depends(get_current_user)):
    sales = await db.sales.find().to_list(5000)
    products = await db.products.find().to_list(2000)
    def ts(p):
        return p.get("stock_store", 0) + p.get("stock_deposit", 0)
    today = datetime.now(timezone.utc).date().isoformat()
    today_sales = [s for s in sales if (s.get("created_at", "")[:10] == today)]
    return {"today_total": round(sum(s.get("total", 0) for s in today_sales), 2), "today_count": len(today_sales),
            "total_revenue": round(sum(s.get("total", 0) for s in sales), 2),
            "total_profit": round(sum(s.get("profit", 0) for s in sales), 2), "product_count": len(products),
            "low_stock": sum(1 for p in products if 0 < ts(p) <= p.get("min_stock", 5)),
            "out_of_stock": sum(1 for p in products if ts(p) <= 0),
            "recent": sorted([{"receipt_no": s.get("receipt_no"), "seller_name": s.get("seller_name"),
                               "client_name": s.get("client_name"), "total": s.get("total"),
                               "created_at": s.get("created_at")} for s in sales],
                             key=lambda x: x["created_at"], reverse=True)[:8]}


# ------------------------------------------------------------------ Settings
@api_router.get("/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    s = await db.settings.find_one({"_id": "general"})
    if not s:
        s = {"_id": "general", "defaultTheme": "light", "defaultMargin": 30.0, "commissionRate": 5.0,
             "companyName": "Superion Pro", "companyDoc": "", "companyPhone": "", "companyAddress": "", "logo": ""}
        await db.settings.insert_one(s)
    s.pop("_id", None)
    return s


@api_router.put("/settings")
async def update_settings(data: SettingsIn, user: dict = Depends(require_permission("settings"))):
    update = {k: v for k, v in data.model_dump().items() if v is not None}
    await db.settings.update_one({"_id": "general"}, {"$set": update}, upsert=True)
    s = await db.settings.find_one({"_id": "general"})
    s.pop("_id", None)
    return s


@api_router.get("/public/branding")
async def public_branding():
    s = await db.settings.find_one({"_id": "general"}) or {}
    return {"defaultTheme": s.get("defaultTheme", "light"),
            "companyName": s.get("companyName", "Superion Pro"),
            "tradeName": s.get("tradeName", ""),
            "logo": s.get("logo", ""), "logoLight": s.get("logoLight", ""),
            "logoDark": s.get("logoDark", ""), "logoReceipt": s.get("logoReceipt", ""),
            "colorPrimary": s.get("colorPrimary", "#DC2626"),
            "colorSecondary": s.get("colorSecondary", "#7C4A2D"),
            "colorBackground": s.get("colorBackground", "#FFFFFF"),
            "colorText": s.get("colorText", "#1F2937")}


# ------------------------------------------------------------------ SaaS: invites, registration, billing
class InviteCreate(BaseModel):
    ttl_hours: Optional[int] = 24
    expires_at: Optional[str] = None


class RegisterInput(BaseModel):
    token: str
    company_name: str
    name: str
    username: str
    password: str


def _origin_from_request(request: Request) -> str:
    o = request.headers.get("origin")
    if o:
        return o.rstrip("/")
    ref = request.headers.get("referer", "")
    return ref.split("/register")[0].rstrip("/") if ref else ""


@api_router.post("/admin/invites")
async def create_invite(data: InviteCreate, request: Request, user: dict = Depends(require_superadmin)):
    now = datetime.now(timezone.utc)
    if data.expires_at:
        try:
            exp = datetime.fromisoformat(data.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="Data de expiração inválida")
    else:
        exp = now + timedelta(hours=int(data.ttl_hours or 24))
    token = secrets.token_urlsafe(24)
    doc = {"token": token, "created_at": now.isoformat(), "expires_at": exp.isoformat(),
           "used": False, "used_at": None, "created_by": user["name"], "tenant_id": None}
    await cdb.invites.insert_one(doc)
    origin = _origin_from_request(request)
    return {"token": token, "url": f"{origin}/register?token={token}", "expires_at": exp.isoformat(),
            "used": False, "created_at": doc["created_at"]}


@api_router.get("/admin/invites")
async def list_invites(user: dict = Depends(require_superadmin)):
    now = datetime.now(timezone.utc)
    out = []
    for i in await cdb.invites.find().sort("created_at", -1).to_list(200):
        try:
            expired = now > datetime.fromisoformat(i["expires_at"])
        except Exception:
            expired = False
        out.append({"token": i["token"], "expires_at": i["expires_at"], "created_at": i.get("created_at"),
                    "used": i.get("used", False), "used_at": i.get("used_at"),
                    "expired": expired, "company_name": i.get("company_name")})
    return out


@api_router.get("/invites/{token}")
async def validate_invite(token: str):
    i = await cdb.invites.find_one({"token": token})
    if not i:
        return {"valid": False, "reason": "not_found"}
    if i.get("used"):
        return {"valid": False, "reason": "used"}
    try:
        if datetime.now(timezone.utc) > datetime.fromisoformat(i["expires_at"]):
            return {"valid": False, "reason": "expired"}
    except Exception:
        return {"valid": False, "reason": "expired"}
    return {"valid": True, "expires_at": i["expires_at"]}


@api_router.post("/auth/register")
async def register(data: RegisterInput):
    i = await cdb.invites.find_one({"token": data.token})
    now = datetime.now(timezone.utc)
    if not i or i.get("used"):
        raise HTTPException(status_code=400, detail="Convite inválido ou já utilizado")
    try:
        exp_ok = now <= datetime.fromisoformat(i["expires_at"])
    except Exception:
        exp_ok = False
    if not exp_ok:
        raise HTTPException(status_code=400, detail="Convite expirado")
    uname = data.username.lower().strip()
    if await cdb.users.find_one({"username": uname}):
        raise HTTPException(status_code=400, detail="Nome de usuário já existe")
    trial_end = now + timedelta(days=TRIAL_DAYS)
    base_slug = slugify(data.company_name)
    slug = base_slug
    n = 1
    while await cdb.tenants.find_one({"slug": slug}):
        n += 1
        slug = f"{base_slug}-{n}"
    tres = await cdb.tenants.insert_one({"company_name": data.company_name, "is_primary": False,
        "status": "in_trial", "trial_start": now.isoformat(), "trial_end": trial_end.isoformat(),
        "slug": slug, "created_at": now.isoformat()})
    tid = str(tres.inserted_id)
    ures = await cdb.users.insert_one({"name": data.name, "username": uname,
        "password_hash": hash_password(data.password), "role": "admin", "permissions": ALL_PERMISSIONS,
        "active": True, "created_at": now.isoformat(), "tenant_id": tid})
    await cdb.invites.update_one({"_id": i["_id"]}, {"$set": {"used": True, "used_at": now.isoformat(),
        "tenant_id": tid, "company_name": data.company_name}})
    tdb = tenant_db_for({"_id": tres.inserted_id, "is_primary": False})
    await tdb.settings.insert_one({"_id": "general", "defaultTheme": "light", "defaultMargin": 30.0,
        "commissionRate": 5.0, "companyName": data.company_name, "tradeName": "", "companyDoc": "",
        "companyPhone": "", "companyAddress": "", "logo": "",
        "receiptFooter": "Obrigado pela preferência! Volte sempre."})
    token = create_token(str(ures.inserted_id))
    return {"token": token, "user": {"id": str(ures.inserted_id), "name": data.name, "username": uname,
            "role": "admin", "permissions": ALL_PERMISSIONS, "tenant_id": tid}}


@api_router.get("/billing/status")
async def billing_status(user: dict = Depends(get_current_user)):
    t = user.get("_tenant") or {}
    if t.get("is_primary"):
        return {"status": "paid_active", "is_primary": True, "trial_days_left": None, "company_name": "Matriz"}
    status = t.get("status", "in_trial")
    days_left = None
    if status == "in_trial" and t.get("trial_end"):
        try:
            secs = (datetime.fromisoformat(t["trial_end"]) - datetime.now(timezone.utc)).total_seconds()
            days_left = max(0, int((secs + 86399) // 86400))
        except Exception:
            days_left = None
    return {"status": status, "is_primary": False, "trial_days_left": days_left,
            "trial_end": t.get("trial_end"), "company_name": t.get("company_name", "")}


class CheckoutIn(BaseModel):
    method: str = "pix"
    plan: str = "mensal"


@api_router.post("/billing/checkout")
async def billing_checkout(data: CheckoutIn, user: dict = Depends(get_current_user)):
    t = user.get("_tenant") or {}
    if t.get("is_primary"):
        raise HTTPException(status_code=400, detail="Conta matriz não requer pagamento")
    payment_id = "MP-" + secrets.token_hex(8).upper()
    amount = 149.90
    resp = {"payment_id": payment_id, "status": "pending", "method": data.method,
            "amount": amount, "mocked": True}
    if data.method == "pix":
        resp["pix_copia_cola"] = "00020126" + secrets.token_hex(24) + "5204000053039865802BR"
    await cdb.tenants.update_one({"_id": t["_id"]}, {"$set": {"last_payment_id": payment_id,
        "last_payment_amount": amount}})
    return resp


class ConfirmIn(BaseModel):
    payment_id: str


@api_router.post("/billing/confirm")
async def billing_confirm(data: ConfirmIn, user: dict = Depends(get_current_user)):
    t = user.get("_tenant") or {}
    if t.get("is_primary"):
        return {"status": "paid_active"}
    now = datetime.now(timezone.utc)
    await cdb.tenants.update_one({"_id": t["_id"]}, {"$set": {"status": "paid_active",
        "paid_at": now.isoformat(), "paid_payment_id": data.payment_id}})
    return {"status": "paid_active", "paid_at": now.isoformat()}


# ------------------------------------------------------------------ AI Assistant
ASSISTANT_SYSTEM = """Você é o assistente do sistema ERP/PDV Superion Pro (português do Brasil).
Interprete o comando do usuário e responda SOMENTE com um JSON válido, sem markdown, com o formato:
{"action": "<ação>", "value": <valor ou null>, "reply": "<resposta curta amigável em pt-BR>"}

Ações possíveis:
- "set_theme": value = "light" ou "dark"
- "set_margin": value = número (ex: "mudar margem de lucro para 35%")
- "navigate": value = uma de ["dashboard","pos","products","reports","warehouse","invoices","commissions","service-orders","users","settings"]
- "update_product": value = objeto {"query": "<nome ou EAN do produto a corrigir>", "fields": {<campos a alterar>}}
    campos permitidos em "fields": name, ean, brand, category, cost (número), margin (número %),
    price (número, preço manual), stock_store (número), stock_deposit (número), min_stock (número).
    Use esta ação para QUALQUER correção/edição de cadastro de produto pedida pelo usuário
    (ex: "corrigir o preço do coca cola para 7,50", "mudar a marca do produto X para Nestlé",
    "ajustar estoque da loja do arroz para 20"). Converta valores monetários para número decimal com ponto.
- "info": value = null (apenas conversa/resposta informativa)

Escolha a ação mais adequada. Sempre inclua um campo "reply" curto confirmando a ação."""


@api_router.post("/assistant/command")
async def assistant_command(data: AssistantIn, user: dict = Depends(get_current_user)):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"assist-{user['_id']}",
                       system_message=ASSISTANT_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        resp = await chat.send_message(UserMessage(text=data.message))
        text = (resp if isinstance(resp, str) else str(resp)).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
    except Exception as e:
        logger.warning(f"assistant parse failed: {e}")
        parsed = {"action": "info", "value": None, "reply": "Desculpe, não entendi o comando. Pode reformular?"}
    action = parsed.get("action")
    value = parsed.get("value")
    if action == "set_theme" and value in ("light", "dark") and has_permission(user, "settings"):
        await db.settings.update_one({"_id": "general"}, {"$set": {"defaultTheme": value}}, upsert=True)
    if action == "set_margin" and has_permission(user, "settings"):
        try:
            await db.settings.update_one({"_id": "general"}, {"$set": {"defaultMargin": float(value)}}, upsert=True)
        except Exception:
            pass
    if action == "update_product":
        if not has_permission(user, "products"):
            parsed["reply"] = "Você não tem permissão para editar produtos."
        else:
            try:
                q = (value or {}).get("query", "")
                fields = (value or {}).get("fields", {}) or {}
                prod = None
                if q:
                    prod = await db.products.find_one({"ean": str(q).strip()})
                    if not prod:
                        prod = await db.products.find_one({"name": {"$regex": re.escape(str(q)), "$options": "i"}})
                if not prod:
                    parsed["reply"] = f"Produto '{q}' não encontrado no cadastro."
                else:
                    allowed = {"name", "ean", "brand", "category", "cost", "margin",
                               "price", "stock_store", "stock_deposit", "min_stock"}
                    num_fields = {"cost", "margin", "price", "stock_store", "stock_deposit", "min_stock"}
                    upd = {}
                    for k, v in fields.items():
                        if k not in allowed or v is None:
                            continue
                        upd[k] = float(v) if k in num_fields else str(v)
                    if not upd:
                        parsed["reply"] = "Nenhum campo válido para atualizar foi informado."
                    else:
                        merged = {**prod, **upd}
                        upd["price"] = compute_price(merged.get("cost", 0), merged.get("cost_qty", 1),
                                                     merged.get("margin", 30),
                                                     upd.get("price", prod.get("price")) if "price" in fields else None)
                        await db.products.update_one({"_id": prod["_id"]}, {"$set": upd})
                        parsed["reply"] = f"Produto '{prod.get('name')}' atualizado com sucesso."
            except Exception as e:
                logger.warning(f"assistant update_product failed: {e}")
                parsed["reply"] = "Não consegui aplicar a correção no produto."
    return parsed


# ------------------------------------------------------------------ AI Vision (product recognition)
VISION_SYSTEM = """Você é um analista de produtos de varejo. Recebe a foto de um produto, rótulo,
prateleira ou trecho de nota fiscal e deve extrair os dados do produto principal visível.
Responda SOMENTE com um JSON válido (sem markdown), no formato:
{"name": "<nome/título do produto>", "brand": "<marca/fabricante>", "category": "<categoria>",
 "unit": "<tamanho/embalagem, ex: 500ml, 1kg, pacote 12un>", "price": <preço numérico visível ou null>,
 "details": "<outros detalhes de texto visíveis, curto>"}
Use português do Brasil. Se algum campo não for identificável, use string vazia (ou null para price)."""


class VisionIn(BaseModel):
    image_base64: str


@api_router.post("/vision/analyze-product")
async def analyze_product_image(data: VisionIn, user: dict = Depends(get_current_user),
                                _gate: dict = Depends(require_active_tenant)):
    b64 = (data.image_base64 or "").strip()
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    if not b64:
        raise HTTPException(status_code=400, detail="Imagem vazia")
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"vision-{user['_id']}-{uuid.uuid4().hex[:6]}",
                       system_message=VISION_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        resp = await chat.send_message(UserMessage(
            text="Analise esta imagem e extraia os dados do produto no formato JSON solicitado.",
            file_contents=[ImageContent(image_base64=b64)]))
        text = (resp if isinstance(resp, str) else str(resp)).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return {"found": True, "name": parsed.get("name", ""), "brand": parsed.get("brand", ""),
                "category": parsed.get("category", ""), "unit": parsed.get("unit", ""),
                "price": parsed.get("price"), "details": parsed.get("details", "")}
    except Exception as e:
        logger.warning(f"vision analyze failed: {e}")
        return {"found": False, "name": "", "brand": "", "category": "", "unit": "",
                "price": None, "details": "", "error": "Não foi possível analisar a imagem."}


# ------------------------------------------------------------------ AI Shelf / Gondola multi-product scanner
SHELF_SYSTEM = """Você é um sistema de visão computacional para varejo. Recebe uma ou mais imagens
(foto de prateleira/gôndola, rótulo de produto, ou páginas de um PDF/nota fiscal) e deve DETECTAR
e EXTRAIR todos os produtos DISTINTOS visíveis.
Responda SOMENTE com JSON válido (sem markdown), no formato:
{"items": [
  {"name": "<nome/título do produto>", "brand": "<marca/fabricante>", "pack_size": "<tamanho/embalagem ex: 500ml, 1kg>",
   "price": <preço numérico visível na etiqueta ou null>, "ean": "<código de barras se legível ou vazio>",
   "category": "<categoria estimada>", "confidence": <0.0 a 1.0 de confiança da detecção>}
]}
Regras: use português do Brasil; agrupe itens idênticos em uma única entrada; não invente dados —
se não conseguir ler, deixe vazio/null. Considere qualquer instrução adicional do usuário fornecida no texto."""


class ShelfScanIn(BaseModel):
    file_base64: str
    mime_type: str = "image/jpeg"
    note: str = ""


def _pdf_to_images_b64(pdf_bytes: bytes, max_pages: int = 6):
    import fitz
    out = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=150)
            out.append(base64.b64encode(pix.tobytes("png")).decode())
    finally:
        doc.close()
    return out


def _off_search_sync(term: str):
    try:
        r = requests.get("https://world.openfoodfacts.org/cgi/search.pl",
                         params={"search_terms": term, "search_simple": 1, "json": 1, "page_size": 1},
                         timeout=6, headers={"User-Agent": "SuperionPro/1.0"})
        prods = r.json().get("products", [])
        if prods:
            p = prods[0]
            return {"ean": p.get("code", "") or "",
                    "image": p.get("image_front_url") or p.get("image_url") or "",
                    "brand": (p.get("brands") or "").split(",")[0].strip()}
    except Exception:
        pass
    return None


@api_router.post("/vision/scan-shelf")
async def scan_shelf(data: ShelfScanIn, user: dict = Depends(get_current_user),
                     _gate: dict = Depends(require_active_tenant)):
    raw = (data.file_base64 or "").strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio")
    try:
        file_bytes = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo inválido")

    if "pdf" in (data.mime_type or "").lower():
        try:
            images_b64 = _pdf_to_images_b64(file_bytes)
        except Exception as e:
            logger.warning(f"pdf render failed: {e}")
            raise HTTPException(status_code=400, detail="Não foi possível ler o PDF")
        if not images_b64:
            raise HTTPException(status_code=400, detail="PDF sem páginas legíveis")
    else:
        images_b64 = [raw]

    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
        note = (data.note or "").strip()
        prompt = "Detecte e liste TODOS os produtos distintos visíveis na(s) imagem(ns)."
        if note:
            prompt += f"\n\nInstrução adicional do usuário (aplique na extração): {note}"
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"shelf-{user['_id']}-{uuid.uuid4().hex[:6]}",
                       system_message=SHELF_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        resp = await asyncio.wait_for(chat.send_message(UserMessage(
            text=prompt, file_contents=[ImageContent(image_base64=b) for b in images_b64])), timeout=120)
        text = (resp if isinstance(resp, str) else str(resp)).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        items = parsed.get("items", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
    except Exception as e:
        logger.warning(f"shelf scan failed: {e}")
        return {"count": 0, "items": [], "note": (data.note or "").strip(),
                "error": "Não foi possível analisar o arquivo."}

    all_products = await db.products.find().to_list(3000)
    results = []
    for it in items[:40]:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        brand = (it.get("brand") or "").strip()
        ean = str(it.get("ean") or "").strip()
        existing = None
        if ean:
            existing = next((p for p in all_products if p.get("ean") == ean), None)
        if not existing:
            nl = name.lower()
            existing = next((p for p in all_products if p.get("name", "").lower() == nl), None)
            if not existing:
                existing = next((p for p in all_products
                                 if nl and (nl in p.get("name", "").lower() or p.get("name", "").lower() in nl)), None)
        results.append({
            "name": name, "brand": brand, "pack_size": it.get("pack_size", "") or "",
            "price": it.get("price"),
            "ean": ean or (existing.get("ean", "") if existing else ""),
            "image": it.get("image", "") or (existing.get("image", "") if existing else ""),
            "category": (it.get("category") or "") or (existing.get("category", "") if existing else ""),
            "confidence": it.get("confidence", 0.8),
            "already_registered": bool(existing),
            "existing_id": str(existing["_id"]) if existing else None,
            "existing_stock": round(existing.get("stock_store", 0) + existing.get("stock_deposit", 0), 3) if existing else 0,
        })

    # Enrich items without EAN via Open Food Facts text search (concurrent)
    idxs = [i for i, r in enumerate(results) if not r["ean"] and not r["already_registered"]][:20]
    if idxs:
        offs = await asyncio.gather(*[
            asyncio.to_thread(_off_search_sync, f'{results[i]["name"]} {results[i]["brand"]}'.strip())
            for i in idxs])
        for i, off in zip(idxs, offs):
            if off:
                results[i]["ean"] = results[i]["ean"] or off["ean"]
                results[i]["image"] = results[i]["image"] or off["image"]
                results[i]["brand"] = results[i]["brand"] or off["brand"]

    return {"count": len(results), "items": results, "note": (data.note or "").strip(), "pages": len(images_b64)}


class BulkItem(BaseModel):
    name: str
    brand: str = ""
    ean: str = ""
    category: str = ""
    image: str = ""
    cost: float = 0.0
    margin: float = 30.0
    margin_wholesale: float = 15.0
    price: Optional[float] = None
    stock_store: float = 0.0
    stock_deposit: float = 0.0
    existing_id: Optional[str] = None


class BulkIn(BaseModel):
    items: List[BulkItem]


@api_router.post("/products/bulk")
async def bulk_products(data: BulkIn, user: dict = Depends(require_permission("products")),
                        _gate: dict = Depends(require_active_tenant)):
    created, updated, skipped = 0, 0, 0
    for it in data.items:
        store_inc = max(0, it.stock_store)
        dep_inc = max(0, it.stock_deposit)
        if it.existing_id:
            if not ObjectId.is_valid(it.existing_id):
                skipped += 1
                continue
            await db.products.update_one(
                {"_id": ObjectId(it.existing_id)},
                {"$inc": {"stock_store": store_inc, "stock_deposit": dep_inc}})
            updated += 1
        else:
            doc = {"name": it.name, "brand": it.brand, "ean": it.ean, "code": "",
                   "category": it.category, "image": it.image, "cost": it.cost, "cost_qty": 1.0,
                   "margin": it.margin, "margin_wholesale": it.margin_wholesale, "unit": "Unidade",
                   "stock_store": it.stock_store, "stock_deposit": it.stock_deposit, "min_stock": 5,
                   "ncm": "", "cest": "", "origem": "0", "cfop": "5102", "csosn": "102",
                   "pis_cst": "07", "cofins_cst": "07"}
            doc["price"] = compute_price(it.cost, 1.0, it.margin, it.price)
            doc["price_wholesale"] = compute_price(it.cost, 1.0, it.margin_wholesale, None)
            doc["stock_store"] = store_inc
            doc["stock_deposit"] = dep_inc
            doc["active"] = True
            doc["created_at"] = now_iso()
            await db.products.insert_one(doc)
            created += 1
    return {"created": created, "updated": updated, "skipped": skipped, "total": created + updated}


# ------------------------------------------------------------------ Reports
@api_router.get("/reports/overview")
async def reports_overview(user: dict = Depends(get_current_user)):
    products = await db.products.find().to_list(5000)
    sales = await db.sales.find().to_list(5000)
    deleted = await db.deleted_products.find().sort("deleted_at", -1).to_list(1000)

    def stock_of(p):
        return p.get("stock_store", 0) + p.get("stock_deposit", 0)

    total_products = len(products)
    stock_qty = round(sum(stock_of(p) for p in products), 3)
    stock_value_cost = round(sum(stock_of(p) * p.get("cost", 0) for p in products), 2)
    stock_value_retail = round(sum(stock_of(p) * p.get("price", 0) for p in products), 2)
    low = sum(1 for p in products if 0 < stock_of(p) <= p.get("min_stock", 5))
    out = sum(1 for p in products if stock_of(p) <= 0)

    sales_count = len(sales)
    revenue = round(sum(s.get("total", 0) for s in sales), 2)
    profit = round(sum(s.get("profit", 0) for s in sales), 2)
    qty_sold = 0.0
    sold_map = {}
    for s in sales:
        for it in s.get("items", []):
            q = it.get("quantity", 0)
            qty_sold += q
            name = it.get("name", "?")
            e = sold_map.setdefault(name, {"name": name, "quantity": 0.0, "revenue": 0.0})
            e["quantity"] += q
            e["revenue"] += it.get("price", 0) * q
    top_products = sorted(sold_map.values(), key=lambda x: x["quantity"], reverse=True)[:20]
    for e in top_products:
        e["quantity"] = round(e["quantity"], 3)
        e["revenue"] = round(e["revenue"], 2)

    deleted_out = [{"name": d.get("name", ""), "ean": d.get("ean", ""), "category": d.get("category", ""),
                    "stock": d.get("stock", 0), "deleted_by": d.get("deleted_by", ""),
                    "deleted_at": d.get("deleted_at", "")} for d in deleted]

    return {
        "metrics": {"total_products": total_products, "stock_qty": stock_qty,
                    "stock_value_cost": stock_value_cost, "stock_value_retail": stock_value_retail,
                    "low_stock": low, "out_of_stock": out, "sales_count": sales_count,
                    "revenue": revenue, "profit": profit, "qty_sold": round(qty_sold, 3),
                    "deleted_count": len(deleted_out)},
        "top_products": top_products, "deleted_products": deleted_out,
    }


def _logo_flowable(settings, max_w=130, max_h=46):
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.utils import ImageReader
    raw = settings.get("logoReceipt") or settings.get("logoLight") or settings.get("logo") or ""
    if not raw or not raw.startswith("data:") or "," not in raw:
        return None
    try:
        img_bytes = base64.b64decode(raw.split(",", 1)[1])
        bio = io.BytesIO(img_bytes)
        iw, ih = ImageReader(bio).getSize()
        ratio = min(max_w / iw, max_h / ih)
        bio.seek(0)
        return RLImage(bio, width=iw * ratio, height=ih * ratio)
    except Exception as e:
        logger.warning(f"pdf logo failed: {e}")
        return None


def build_price_list_pdf(products, settings, rep_name, filters):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=16 * mm, title="Tabela de Preços")
    styles = getSampleStyleSheet()
    small = ParagraphStyle("sm", parent=styles["Normal"], fontSize=8, leading=10)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=9)
    elems = []

    company = settings.get("companyName") or settings.get("tradeName") or "Superion Pro"
    cnpj = settings.get("companyDoc") or "-"
    now = datetime.now(timezone.utc) - timedelta(hours=3)
    right = Paragraph(
        f"<b>{company}</b><br/>CNPJ: {cnpj}<br/>Emitido: {now.strftime('%d/%m/%Y %H:%M')}<br/>Responsável: {rep_name}",
        small)
    logo = _logo_flowable(settings)
    left = logo if logo else Paragraph(f"<b>{company}</b>", styles["Title"])
    header = Table([[left, right]], colWidths=[150, None])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elems += [header, Spacer(1, 8), Paragraph("<b>TABELA DE PREÇOS</b>", styles["Heading2"])]

    fdesc = []
    if filters.get("category"):
        fdesc.append(f"Categoria: {filters['category']}")
    if filters.get("in_stock_only"):
        fdesc.append("Somente itens em estoque")
    if fdesc:
        elems.append(Paragraph("Filtros aplicados — " + " · ".join(fdesc), small))
    elems.append(Spacer(1, 6))

    rows = [["EAN", "Produto", "Categoria", "Estoque", "Atacado", "Varejo"]]
    for p in products:
        stock = round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 2)
        rows.append([p.get("ean", "") or "-", Paragraph(p.get("name", ""), cell),
                     p.get("category", "") or "-", f"{stock:g}",
                     f"R$ {p.get('price_wholesale', p.get('price', 0)) or 0:.2f}",
                     f"R$ {p.get('price', 0) or 0:.2f}"])
    tbl = Table(rows, colWidths=[70, None, 80, 42, 56, 56], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DC2626")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (3, 0), (5, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elems += [tbl, Spacer(1, 28),
              Paragraph("______________________________________", small),
              Paragraph(f"{rep_name} — Responsável", small)]
    doc.build(elems)
    return buf.getvalue()


@api_router.get("/reports/price-list.pdf")
async def price_list_pdf(category: str = "", in_stock_only: bool = False,
                         user: dict = Depends(get_current_user)):
    query = {}
    if category:
        query["category"] = category
    products = await db.products.find(query).sort("name", 1).to_list(5000)
    if in_stock_only:
        products = [p for p in products if (p.get("stock_store", 0) + p.get("stock_deposit", 0)) > 0]
    settings = await db.settings.find_one({"_id": "general"}) or {}
    pdf = build_price_list_pdf(products, settings, user["name"],
                               {"category": category, "in_stock_only": in_stock_only})
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=tabela_precos.pdf"})


# ------------------------------------------------------------------ Commercial: WhatsApp, Catalog, Orders, Gateways
import random as _random

ORDER_STATUSES = ["pending_payment", "paid_preparing", "ready", "out_for_delivery", "completed"]


def slugify(name: str) -> str:
    import re as _re
    import unicodedata
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = _re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "loja"


def gen_pix(amount: float, pix_key: str = "pix@superionpro.app", merchant: str = "SUPERION PRO") -> dict:
    txid = secrets.token_hex(12)
    amt = f"{amount:.2f}"
    payload = (f"00020126360014BR.GOV.BCB.PIX0114{pix_key}520400005303986540{len(amt)}{amt}"
               f"5802BR5913{merchant[:13]:<13}6009SAO PAULO62070503{txid[:6]}6304{txid[:4].upper()}")
    return {"pix_copia_cola": payload, "amount": round(amount, 2), "txid": txid}


def order_out(o: dict) -> dict:
    o["id"] = str(o.pop("_id"))
    return o


async def _deduct_stock(items):
    for it in items:
        pid = it.get("product_id")
        if pid and ObjectId.is_valid(pid):
            await db.products.update_one({"_id": ObjectId(pid)}, {"$inc": {"stock_store": -it.get("quantity", 0)}})


async def _pix_key_for_tenant():
    s = await db.settings.find_one({"_id": "general"}) or {}
    gw = s.get("payment_gateway") or {}
    return gw.get("pix_key") or "pix@superionpro.app", s.get("companyName") or "SUPERION PRO"


async def _create_order(source, customer, items, status="pending_payment", note="", external_id=None):
    sub = round(sum(i["price"] * i["quantity"] for i in items), 2)
    pix_key, merchant = await _pix_key_for_tenant()
    pix = gen_pix(sub, pix_key, merchant)
    paid = status != "pending_payment"
    doc = {"source": source, "customer": customer, "items": items, "subtotal": sub, "total": sub,
           "status": status, "note": note, "external_id": external_id,
           "payment": {"method": "pix", "pix_copia_cola": pix["pix_copia_cola"], "paid": paid},
           "stock_deducted": False, "created_at": now_iso(), "updated_at": now_iso()}
    if paid:
        await _deduct_stock(items)
        doc["stock_deducted"] = True
    res = await db.orders.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    return doc


# ---- Orders / Kanban ----
class OrderItemIn(BaseModel):
    product_id: Optional[str] = None
    name: str
    quantity: float = 1
    price: float = 0.0


class OrderIn(BaseModel):
    source: str = "whatsapp"
    customer_name: str = ""
    customer_phone: str = ""
    customer_address: str = ""
    items: List[OrderItemIn] = []
    note: str = ""


@api_router.get("/orders")
async def list_orders(source: str = "", user: dict = Depends(get_current_user)):
    q = {} if not source else {"source": source}
    orders = await db.orders.find(q).sort("created_at", -1).to_list(1000)
    return [order_out(o) for o in orders]


@api_router.post("/orders")
async def create_order(data: OrderIn, user: dict = Depends(get_current_user)):
    customer = {"name": data.customer_name, "phone": data.customer_phone, "address": data.customer_address}
    items = [i.model_dump() for i in data.items]
    return await _create_order(data.source, customer, items, "pending_payment", data.note)


@api_router.put("/orders/{oid}/status")
async def update_order_status(oid: str, body: dict, user: dict = Depends(get_current_user)):
    status = body.get("status")
    if status not in ORDER_STATUSES:
        raise HTTPException(status_code=400, detail="Status inválido")
    o = await db.orders.find_one({"_id": ObjectId(oid)})
    if not o:
        raise HTTPException(status_code=404, detail="Pedido não encontrado")
    upd = {"status": status, "updated_at": now_iso()}
    if status != "pending_payment" and not o.get("stock_deducted"):
        await _deduct_stock(o.get("items", []))
        upd["stock_deducted"] = True
        upd["payment.paid"] = True
    await db.orders.update_one({"_id": o["_id"]}, {"$set": upd})
    return order_out(await db.orders.find_one({"_id": o["_id"]}))


@api_router.post("/orders/simulate")
async def simulate_order(body: dict, user: dict = Depends(get_current_user)):
    source = body.get("source", "ifood")
    if source not in ("ifood", "99food", "whatsapp", "catalog"):
        source = "ifood"
    prods = await db.products.find({"$expr": {"$gt": [{"$add": ["$stock_store", "$stock_deposit"]}, 0]}}).to_list(200)
    if not prods:
        raise HTTPException(status_code=400, detail="Nenhum produto em estoque para simular um pedido pago")
    chosen = _random.sample(prods, min(len(prods), _random.randint(1, 3)))
    items = [{"product_id": str(p["_id"]), "name": p.get("name", ""), "quantity": _random.randint(1, 3),
              "price": p.get("price", 0)} for p in chosen]
    names = ["João Silva", "Maria Souza", "Carlos Lima", "Ana Costa", "Pedro Alves", "Juliana Rocha"]
    customer = {"name": _random.choice(names), "phone": "5511" + str(_random.randint(900000000, 999999999)),
                "address": f"Rua das Flores, {_random.randint(1, 999)}"}
    status = "paid_preparing" if source in ("ifood", "99food") else "pending_payment"
    ext = source.upper() + "-" + secrets.token_hex(4).upper()
    return await _create_order(source, customer, items, status, "Pedido simulado", ext)


# ---- Delivery webhooks (public; tenant resolved by slug) ----
@api_router.post("/webhooks/{provider}/{slug}")
async def delivery_webhook(provider: str, slug: str, payload: dict, request: Request):
    secret = os.environ.get("DELIVERY_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="Webhook não configurado")
    provided = request.headers.get("x-webhook-secret", "")
    if not provided or not hmac.compare_digest(provided, secret):
        raise HTTPException(status_code=401, detail="Assinatura de webhook inválida")
    t = await cdb.tenants.find_one({"slug": slug})
    if not t:
        raise HTTPException(status_code=404, detail="Tenant não encontrado")
    _current_tenant_db.set(tenant_db_for(t))
    src = "ifood" if "ifood" in provider else "99food"
    norm = await _server_priced_items(payload.get("items", []))
    if not norm:
        raise HTTPException(status_code=400, detail="Nenhum item válido no pedido")
    customer = {"name": payload.get("customer_name", "Cliente"), "phone": payload.get("phone", ""),
                "address": payload.get("address", "")}
    return await _create_order(src, customer, norm, "paid_preparing", "Via webhook", payload.get("external_id"))


# ---- WhatsApp (simulated pairing + bot engine) ----
class WaConnectIn(BaseModel):
    phone: str


@api_router.get("/whatsapp/status")
async def whatsapp_status(user: dict = Depends(get_current_user)):
    s = await db.settings.find_one({"_id": "general"}) or {}
    return s.get("whatsapp") or {"status": "disconnected", "phone": "", "connected_at": None}


@api_router.post("/whatsapp/connect")
async def whatsapp_connect(data: WaConnectIn, user: dict = Depends(require_permission("settings"))):
    qr = "wa-pair:" + secrets.token_hex(24)
    wa = {"status": "generating_qr", "phone": data.phone, "qr": qr, "connected_at": None}
    await db.settings.update_one({"_id": "general"}, {"$set": {"whatsapp": wa}}, upsert=True)
    return wa


@api_router.post("/whatsapp/confirm")
async def whatsapp_confirm(user: dict = Depends(require_permission("settings"))):
    s = await db.settings.find_one({"_id": "general"}) or {}
    wa = s.get("whatsapp") or {}
    wa.update({"status": "connected", "connected_at": now_iso(), "qr": None})
    await db.settings.update_one({"_id": "general"}, {"$set": {"whatsapp": wa}}, upsert=True)
    return wa


@api_router.post("/whatsapp/disconnect")
async def whatsapp_disconnect(user: dict = Depends(require_permission("settings"))):
    wa = {"status": "disconnected", "phone": "", "connected_at": None}
    await db.settings.update_one({"_id": "general"}, {"$set": {"whatsapp": wa}}, upsert=True)
    return wa


def _match_items(text: str, products: list):
    import re as _re
    tl = (text or "").lower()
    found = []
    for p in products:
        name = p.get("name", "")
        key = name.lower().split()[0] if name else ""
        stock = p.get("stock_store", 0) + p.get("stock_deposit", 0)
        if key and len(key) >= 3 and key in tl and stock > 0:
            m = _re.search(r"(\d+)\s*(?:x|un|unid|unidades)?\s*" + _re.escape(key), tl)
            qty = int(m.group(1)) if m else 1
            found.append({"product_id": str(p["_id"]), "name": name, "quantity": qty, "price": p.get("price", 0)})
    return found


class WaBotIn(BaseModel):
    from_phone: str = ""
    from_name: str = "Cliente"
    text: str = ""


# ---- Coupons / Promotions ----
class CouponIn(BaseModel):
    code: str
    discount_pct: float = 0.0
    expiry_date: str = ""
    min_order: float = 0.0
    active: bool = True


def _coupon_out(c):
    c["id"] = str(c.pop("_id"))
    return c


async def _active_coupons():
    today = date.today().isoformat()
    out = []
    for c in await db.coupons.find({"active": True}).sort("discount_pct", -1).to_list(200):
        if c.get("expiry_date") and c["expiry_date"][:10] < today:
            continue
        out.append(c)
    return out


@api_router.get("/coupons")
async def list_coupons(user: dict = Depends(get_current_user)):
    return [_coupon_out(c) for c in await db.coupons.find().sort("created_at", -1).to_list(500)]


@api_router.post("/coupons")
async def create_coupon(data: CouponIn, user: dict = Depends(require_permission("products"))):
    doc = data.model_dump()
    doc["code"] = doc["code"].strip().upper()
    doc["created_at"] = now_iso()
    res = await db.coupons.insert_one(doc)
    await write_audit(user, "CREATE", "Cupons", res.inserted_id, new_values=_json_safe(doc), resource_name=doc["code"])
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    return doc


@api_router.put("/coupons/{cid}")
async def update_coupon(cid: str, data: CouponIn, user: dict = Depends(require_permission("products"))):
    old = await db.coupons.find_one({"_id": ObjectId(cid)}) or {}
    upd = data.model_dump()
    upd["code"] = upd["code"].strip().upper()
    await db.coupons.update_one({"_id": ObjectId(cid)}, {"$set": upd})
    new = await db.coupons.find_one({"_id": ObjectId(cid)})
    keys = list(upd.keys())
    await write_audit(user, "UPDATE", "Cupons", cid, _snap(old, keys), _snap(new, keys), diff_only=True, resource_name=new.get("code", ""))
    return _coupon_out(new)


@api_router.delete("/coupons/{cid}")
async def delete_coupon(cid: str, user: dict = Depends(require_permission("products"))):
    old = await db.coupons.find_one({"_id": ObjectId(cid)}) or {}
    await db.coupons.delete_one({"_id": ObjectId(cid)})
    await write_audit(user, "DELETE", "Cupons", cid, old_values=_json_safe({k: v for k, v in old.items() if k != "_id"}), resource_name=old.get("code", ""))
    return {"ok": True}


# ---- Catalog admin ----
CATALOG_FIELDS = ["catalog_active", "catalog_featured", "catalog_price", "catalog_image", "catalog_description"]


class CatalogProductUpdate(BaseModel):
    catalog_active: Optional[bool] = None
    catalog_featured: Optional[bool] = None
    catalog_price: Optional[float] = None
    catalog_image: Optional[str] = None
    catalog_description: Optional[str] = None


@api_router.get("/catalog/share")
async def catalog_share(request: Request, user: dict = Depends(require_permission("products"))):
    slug = (user.get("_tenant") or {}).get("slug") or await _tenant_slug(user)
    base = _public_base(request)
    url = f"{base}/loja/{slug}" if base else f"https://superionpro.com/loja/{slug}"
    return {"slug": slug, "url": url, "qr_image": _qr_data_url(url)}


@api_router.get("/catalog/admin")
async def catalog_admin(user: dict = Depends(require_permission("products"))):
    prods = await db.products.find().sort("name", 1).to_list(3000)
    out = [{"id": str(p["_id"]), "name": p.get("name", ""), "ean": p.get("ean", ""), "code": p.get("code", ""),
            "category": p.get("category", ""), "image": p.get("catalog_image") or p.get("image", ""),
            "price": p.get("price", 0), "catalog_active": p.get("catalog_active", True),
            "catalog_featured": p.get("catalog_featured", False), "catalog_price": p.get("catalog_price"),
            "catalog_image": p.get("catalog_image", ""), "catalog_description": p.get("catalog_description", ""),
            "stock": round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 2)} for p in prods]
    cats = sorted({p.get("category", "") for p in prods if p.get("category")})
    return {"products": out, "categories": cats,
            "active": sum(1 for p in out if p["catalog_active"]), "hidden": sum(1 for p in out if not p["catalog_active"])}


@api_router.put("/catalog/product/{pid}")
async def catalog_update_product(pid: str, data: CatalogProductUpdate, user: dict = Depends(require_permission("products"))):
    old = await db.products.find_one({"_id": ObjectId(pid)})
    if not old:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    upd = {k: v for k, v in data.model_dump().items() if v is not None}
    if not upd:
        return product_out(old)
    await db.products.update_one({"_id": ObjectId(pid)}, {"$set": upd})
    new = await db.products.find_one({"_id": ObjectId(pid)})
    await write_audit(user, "UPDATE", "Catálogo", pid, _snap(old, CATALOG_FIELDS), _snap(new, CATALOG_FIELDS),
                      diff_only=True, resource_name=new.get("name", ""))
    return product_out(new)


async def _catalog_execute(action, target="", value=None, category="", status=None, actor=None):
    products = await db.products.find().to_list(3000)
    if action in ("disable_catalog_product", "enable_catalog_product", "update_catalog_price"):
        p = _best_product_match(target, products)
        if not p:
            return {"ok": False, "message": f"Não encontrei o produto '{target}' no catálogo."}
        if action == "disable_catalog_product":
            await db.products.update_one({"_id": p["_id"]}, {"$set": {"catalog_active": False}})
            await write_audit(actor, "UPDATE", "Catálogo", str(p["_id"]), {"catalog_active": p.get("catalog_active", True)}, {"catalog_active": False}, resource_name=p["name"])
            return {"ok": True, "message": f"✅ *{p['name']}* foi OCULTADO do catálogo público."}
        if action == "enable_catalog_product":
            await db.products.update_one({"_id": p["_id"]}, {"$set": {"catalog_active": True}})
            await write_audit(actor, "UPDATE", "Catálogo", str(p["_id"]), {"catalog_active": p.get("catalog_active", True)}, {"catalog_active": True}, resource_name=p["name"])
            return {"ok": True, "message": f"✅ *{p['name']}* está ATIVO novamente no catálogo público."}
        try:
            np_ = float(value)
        except Exception:
            return {"ok": False, "message": "Preço inválido."}
        await db.products.update_one({"_id": p["_id"]}, {"$set": {"catalog_price": round(np_, 2)}})
        await write_audit(actor, "UPDATE", "Catálogo", str(p["_id"]), {"catalog_price": p.get("catalog_price")}, {"catalog_price": round(np_, 2)}, resource_name=p["name"])
        return {"ok": True, "message": f"✅ Preço de *{p['name']}* no catálogo atualizado para R$ {np_:.2f}."}
    if action == "bulk_toggle_category":
        active = status in (True, "active", "ativar", "ativo", "on", "ativa")
        cat = (category or target or "").strip()
        matched = [p for p in products if cat and cat.lower() in (p.get("category", "") or "").lower()]
        if not matched:
            return {"ok": False, "message": f"Nenhum produto na categoria '{cat}'."}
        for p in matched:
            await db.products.update_one({"_id": p["_id"]}, {"$set": {"catalog_active": active}})
        await write_audit(actor, "UPDATE", "Catálogo", "", {}, {"categoria": cat, "catalog_active": active, "itens": len(matched)}, resource_name=f"Categoria {cat}")
        verb = "ATIVADA" if active else "PAUSADA"
        return {"ok": True, "message": f"✅ Categoria *{cat}* {verb} no catálogo ({len(matched)} produto(s))."}
    return {"ok": False, "message": "Ação de catálogo não reconhecida."}


CATALOG_AGENT_SYSTEM = """Você interpreta comandos de administração do catálogo digital (pt-BR). Responda SOMENTE JSON válido, sem markdown:
{"action":"disable_catalog_product|enable_catalog_product|update_catalog_price|bulk_toggle_category|answer","target":"<nome ou sku do produto>","value":<novo preço numérico ou null>,"category":"<categoria p/ bulk>","status":"active|paused"}
Exemplos: "desativa o refrigerante" -> disable_catalog_product target refrigerante. "acabou a coxinha, tira da loja" -> disable_catalog_product target coxinha. "coloca a batata frita de volta" -> enable_catalog_product target "batata frita". "muda o preço do arroz para 25,90" -> update_catalog_price target arroz value 25.9. "pausa todos os salgados" -> bulk_toggle_category category salgados status paused. "ativa a categoria bebidas" -> bulk_toggle_category category bebidas status active. Se não for comando de catálogo, use action answer."""


class CatalogAgentIn(BaseModel):
    message: str


@api_router.post("/v1/ai/catalog-agent")
async def catalog_agent(data: CatalogAgentIn, user: dict = Depends(require_permission("products"))):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"cat-{user['_id']}-{uuid.uuid4().hex[:6]}",
                       system_message=CATALOG_AGENT_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        resp = await asyncio.wait_for(chat.send_message(UserMessage(text=data.message)), timeout=60)
        txt = (resp if isinstance(resp, str) else str(resp)).strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            if txt.startswith("json"):
                txt = txt[4:]
        obj = json.loads(txt.strip())
    except Exception as e:
        logger.warning(f"catalog-agent parse failed: {e}")
        return {"ok": False, "reply": "Não entendi o comando de catálogo. Tente: 'desativa o refrigerante do catálogo'."}
    action = obj.get("action")
    if action in (None, "answer"):
        return {"ok": False, "reply": "Diga, por exemplo: 'desativa o refrigerante', 'coloca a coxinha de volta', 'muda o preço do arroz para 25,90' ou 'pausa os salgados'."}
    res = await _catalog_execute(action, obj.get("target", ""), obj.get("value"), obj.get("category", ""), obj.get("status"), user)
    return {"ok": res["ok"], "reply": res["message"]}


GREETING_RE = re.compile(r"^\s*(oi+|ol[áa]|boa\s*(tarde|noite|madrugada)|bom\s*dia|tudo\s*bem|eai|e\s*a[ií]|hey|hello|opa|al[ôo]|menu|in[ií]cio)[\s!?.]*$", re.I)
WELCOME_COOLDOWN_H = 12


def _public_base(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}" if host else ""


async def _build_welcome(request: Request, name: str, slug: str) -> str:
    s = await db.settings.find_one({"_id": "general"}) or {}
    store = s.get("companyName") or s.get("tradeName") or "nossa loja"
    slug = slug or "matriz"
    base = _public_base(request)
    catalog = f"{base}/loja/{slug}" if base else f"https://superionpro.com/loja/{slug}"
    greet = f"Olá, {name}!" if name and name.lower() != "cliente" else "Olá!"
    parts = [f"{greet} 👋 Seja bem-vindo(a) à *{store}*!",
             f"\n🛍️ Confira nosso catálogo digital:\n{catalog}"]
    coupons = await _active_coupons()
    if coupons:
        parts.append("\n🎟️ *Cupons ativos:*")
        for c in coupons[:10]:
            line = f"• `{c['code']}` — {c.get('discount_pct', 0):.0f}% OFF"
            if c.get("min_order"):
                line += f" (pedido mín. R$ {c.get('min_order', 0):.2f})"
            if c.get("expiry_date"):
                try:
                    line += f" · válido até {date.fromisoformat(c['expiry_date'][:10]).strftime('%d/%m/%Y')}"
                except Exception:
                    pass
            parts.append(line)
    parts.append("\n👉 Navegue pelo catálogo, use um cupom ou *digite seu pedido* aqui mesmo (ex.: '2 arroz, 1 feijão'). Estou à disposição! 😊")
    return "\n".join(parts)


@api_router.post("/whatsapp/bot/message")
async def whatsapp_bot(data: WaBotIn, request: Request, user: dict = Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    conv = await db.wa_conversations.find_one({"phone": data.from_phone}) if data.from_phone else None
    products = await db.products.find().to_list(500)
    items = _match_items(data.text, products)

    # First-touch / after-inactivity automated greeting (anti-spam via 12h cooldown)
    is_greeting = bool(GREETING_RE.match(data.text or ""))
    last_welcome = None
    if conv and conv.get("last_welcome_at"):
        try:
            last_welcome = datetime.fromisoformat(conv["last_welcome_at"])
        except Exception:
            last_welcome = None
    cooled = (last_welcome is None) or ((now - last_welcome).total_seconds() > WELCOME_COOLDOWN_H * 3600)
    should_welcome = (not items) and (is_greeting or conv is None) and cooled

    await db.wa_conversations.update_one(
        {"phone": data.from_phone},
        {"$set": {"phone": data.from_phone, "name": data.from_name, "last_message_at": now_iso()},
         "$setOnInsert": {"created_at": now_iso()}}, upsert=True)

    if should_welcome:
        slug = (user.get("_tenant") or {}).get("slug") or await _tenant_slug(user)
        reply = await _build_welcome(request, data.from_name, slug)
        await db.wa_conversations.update_one({"phone": data.from_phone}, {"$set": {"last_welcome_at": now_iso()}})
        return {"matched": False, "welcome": True, "reply": reply}

    if not items:
        return {"matched": False, "reply": "Olá! Não encontrei esses produtos. Envie, por exemplo: '2 arroz, 1 feijao'. "
                "Veja nosso catálogo para os nomes exatos."}
    order = await _create_order("whatsapp",
                                {"name": data.from_name, "phone": data.from_phone, "address": ""},
                                items, "pending_payment", "Pedido via WhatsApp Bot")
    lines = "\n".join([f"• {i['quantity']}x {i['name']} — R$ {i['price']:.2f}" for i in items])
    reply = (f"Pedido recebido, {data.from_name}! \n\n{lines}\n\n*Total: R$ {order['total']:.2f}*\n\n"
             f"Pague com Pix Copia e Cola:\n{order['payment']['pix_copia_cola']}\n\n"
             "Assim que recebermos, começamos a preparar. 🛒")
    return {"matched": True, "reply": reply, "order_id": order["id"],
            "pix": order["payment"]["pix_copia_cola"], "total": order["total"]}


# ---- Merchant payment gateway settings ----
class GatewayIn(BaseModel):
    provider: Optional[str] = None            # mercadopago | asaas | pagbank
    mp_access_token: Optional[str] = None
    asaas_key: Optional[str] = None
    pagbank_token: Optional[str] = None
    pix_key: Optional[str] = None


@api_router.get("/settings/payment-gateway")
async def get_gateway(user: dict = Depends(require_permission("settings"))):
    s = await db.settings.find_one({"_id": "general"}) or {}
    gw = s.get("payment_gateway") or {}
    return {"provider": gw.get("provider", "mercadopago"), "pix_key": gw.get("pix_key", ""),
            "mp_access_token": gw.get("mp_access_token", ""), "asaas_key": gw.get("asaas_key", ""),
            "pagbank_token": gw.get("pagbank_token", ""),
            "connected": bool(gw.get("mp_access_token") or gw.get("asaas_key") or gw.get("pagbank_token"))}


@api_router.put("/settings/payment-gateway")
async def put_gateway(data: GatewayIn, user: dict = Depends(require_permission("settings"))):
    s = await db.settings.find_one({"_id": "general"}) or {}
    gw = s.get("payment_gateway") or {}
    gw.update({k: v for k, v in data.model_dump().items() if v is not None})
    await db.settings.update_one({"_id": "general"}, {"$set": {"payment_gateway": gw}}, upsert=True)
    return {"ok": True, "provider": gw.get("provider"),
            "connected": bool(gw.get("mp_access_token") or gw.get("asaas_key") or gw.get("pagbank_token"))}


# ---- Public Digital Catalog (no auth; tenant by slug) ----
async def _set_ctx_by_slug(slug: str):
    t = await cdb.tenants.find_one({"slug": slug})
    if not t:
        raise HTTPException(status_code=404, detail="Catálogo não encontrado")
    _current_tenant_db.set(tenant_db_for(t))
    return t


@api_router.get("/public/catalog/{slug}")
async def public_catalog(slug: str):
    t = await _set_ctx_by_slug(slug)
    s = await db.settings.find_one({"_id": "general"}) or {}
    wa = s.get("whatsapp") or {}
    prods = await db.products.find({"$expr": {"$gt": [{"$add": ["$stock_store", "$stock_deposit"]}, 0]}}).to_list(1000)
    prods = [p for p in prods if p.get("catalog_active", True)]
    prods.sort(key=lambda p: (not p.get("catalog_featured", False), p.get("name", "").lower()))
    items = [{"id": str(p["_id"]), "name": p.get("name", ""), "brand": p.get("brand", ""),
              "category": p.get("category", ""),
              "image": p.get("catalog_image") or p.get("image", ""),
              "price": p.get("catalog_price") if p.get("catalog_price") else p.get("price", 0),
              "list_price": p.get("price", 0), "featured": bool(p.get("catalog_featured", False)),
              "description": p.get("catalog_description", ""),
              "stock": round(p.get("stock_store", 0) + p.get("stock_deposit", 0), 2)} for p in prods]
    coupons = [{"code": c["code"], "discount_pct": c.get("discount_pct", 0), "min_order": c.get("min_order", 0),
                "expiry_date": c.get("expiry_date", "")} for c in await _active_coupons()]
    return {"company_name": s.get("companyName") or t.get("company_name", "Loja"),
            "logo": s.get("logoLight") or s.get("logo", ""),
            "whatsapp_phone": wa.get("phone") or s.get("companyPhone", ""),
            "slug": slug, "products": items, "coupons": coupons}


class CatalogOrderIn(BaseModel):
    customer_name: str = "Cliente"
    customer_phone: str = ""
    items: List[OrderItemIn] = []


async def _server_priced_items(client_items):
    """Rebuild order items from server-side catalog data; ignore any client-supplied price.
    Only active, in-catalog products are accepted. Prevents price tampering (SEC-002)."""
    products = await db.products.find().to_list(3000)
    by_id = {str(p["_id"]): p for p in products}
    out = []
    for ci in client_items:
        pid = ci.get("product_id")
        name = ci.get("name", "")
        try:
            qty = max(1.0, float(ci.get("quantity") or 1))
        except Exception:
            qty = 1.0
        p = by_id.get(pid) if pid else None
        if not p and name:
            p = _best_product_match(name, products)
        if not p or not p.get("catalog_active", True):
            continue
        price = p.get("catalog_price") or p.get("price", 0)
        out.append({"product_id": str(p["_id"]), "name": p.get("name"), "quantity": qty, "price": round(price, 2)})
    return out


@api_router.post("/public/catalog/{slug}/order")
async def public_catalog_order(slug: str, data: CatalogOrderIn):
    await _set_ctx_by_slug(slug)
    if not data.items:
        raise HTTPException(status_code=400, detail="Carrinho vazio")
    items = await _server_priced_items([i.model_dump() for i in data.items])
    if not items:
        raise HTTPException(status_code=400, detail="Nenhum item válido disponível no catálogo")
    order = await _create_order("catalog",
                                {"name": data.customer_name, "phone": data.customer_phone, "address": ""},
                                items, "pending_payment", "Pedido via Catálogo")
    return {"order_id": order["id"], "total": order["total"], "pix": order["payment"]["pix_copia_cola"]}


# ------------------------------------------------------------------ Suppliers & AI Purchasing Quotation
class SupplierIn(BaseModel):
    name: str
    contact_name: str = ""
    whatsapp: str = ""
    categories: str = ""
    notes: str = ""


class QuoteItemIn(BaseModel):
    name: str
    quantity: float = 1


class QuoteIn(BaseModel):
    title: str = "Cotação de Compras"
    items: List[QuoteItemIn] = []
    supplier_ids: List[str] = []


class QuoteResponseIn(BaseModel):
    supplier_id: str
    text: Optional[str] = None
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _tok(s: str) -> set:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return set(w for w in re.split(r"[^a-z0-9]+", s) if len(w) >= 3)


def _supplier_out(s):
    s["id"] = str(s.pop("_id"))
    return s


@api_router.get("/suppliers")
async def list_suppliers(user: dict = Depends(get_current_user)):
    out = await db.suppliers.find().sort("name", 1).to_list(1000)
    return [_supplier_out(s) for s in out]


@api_router.post("/suppliers")
async def create_supplier(data: SupplierIn, user: dict = Depends(require_permission("products"))):
    doc = data.model_dump()
    doc["created_at"] = now_iso()
    res = await db.suppliers.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    await write_audit(user, "CREATE", "Fornecedores", doc["id"], new_values=_json_safe(data.model_dump()),
                      resource_name=doc.get("name", ""))
    return doc


@api_router.put("/suppliers/{sid}")
async def update_supplier(sid: str, data: SupplierIn, user: dict = Depends(require_permission("products"))):
    old = await db.suppliers.find_one({"_id": ObjectId(sid)}) or {}
    await db.suppliers.update_one({"_id": ObjectId(sid)}, {"$set": data.model_dump()})
    new = await db.suppliers.find_one({"_id": ObjectId(sid)})
    keys = list(data.model_dump().keys())
    await write_audit(user, "UPDATE", "Fornecedores", sid, _snap(old, keys), _snap(new, keys),
                      diff_only=True, resource_name=new.get("name", ""))
    return _supplier_out(new)


@api_router.delete("/suppliers/{sid}")
async def delete_supplier(sid: str, user: dict = Depends(require_permission("products"))):
    old = await db.suppliers.find_one({"_id": ObjectId(sid)}) or {}
    await db.suppliers.delete_one({"_id": ObjectId(sid)})
    await write_audit(user, "DELETE", "Fornecedores", sid,
                      old_values=_json_safe({k: v for k, v in old.items() if k != "_id"}),
                      resource_name=old.get("name", ""))
    return {"ok": True}


def _quote_out(q):
    q["id"] = str(q.pop("_id"))
    return q


@api_router.get("/quotes")
async def list_quotes(user: dict = Depends(get_current_user)):
    out = await db.quotes.find().sort("created_at", -1).to_list(500)
    return [_quote_out(q) for q in out]


@api_router.get("/quotes/{qid}")
async def get_quote(qid: str, user: dict = Depends(get_current_user)):
    q = await db.quotes.find_one({"_id": ObjectId(qid)})
    if not q:
        raise HTTPException(status_code=404, detail="Cotação não encontrada")
    return _quote_out(q)


@api_router.delete("/quotes/{qid}")
async def delete_quote(qid: str, user: dict = Depends(require_permission("products"))):
    await db.quotes.delete_one({"_id": ObjectId(qid)})
    return {"ok": True}


@api_router.post("/quotes")
async def create_quote(data: QuoteIn, user: dict = Depends(require_permission("products"))):
    suppliers = await db.suppliers.find({"_id": {"$in": [ObjectId(s) for s in data.supplier_ids if ObjectId.is_valid(s)]}}).to_list(500)
    item_lines = "\n".join([f"- {i.quantity}x {i.name}" for i in data.items])
    sent = []
    for s in suppliers:
        msg = (f"Olá {s.get('contact_name') or s.get('name')}! Gostaríamos de uma cotação para:\n\n"
               f"{item_lines}\n\nPode nos enviar os preços? Obrigado!")
        phone = re.sub(r"\D", "", s.get("whatsapp", ""))
        sent.append({"supplier_id": str(s["_id"]), "supplier_name": s.get("name"),
                     "whatsapp": s.get("whatsapp", ""),
                     "wa_link": f"https://wa.me/{phone}?text={quote_plus(msg)}", "message": msg})
    doc = {"title": data.title, "items": [i.model_dump() for i in data.items],
           "supplier_ids": data.supplier_ids, "sent": sent, "responses": {},
           "status": "sent", "created_at": now_iso()}
    res = await db.quotes.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    await write_audit(user, "CREATE", "Cotações", doc["id"], new_values={
        "titulo": data.title, "itens": len(data.items), "fornecedores": len(sent)}, resource_name=data.title)
    return doc


def _parse_price_text(text: str):
    out = []
    for line in (text or "").splitlines():
        m = re.search(r"(.+?)[\s:：>–\-]+(?:r\$\s*)?(\d{1,4}(?:[.,]\d{3})*[.,]\d{2}|\d{1,4})\s*$", line.strip(), re.I)
        if m:
            name = m.group(1).strip(" .:-–\t")
            raw = m.group(2).replace(".", "").replace(",", ".") if ("," in m.group(2)) else m.group(2)
            try:
                price = float(raw)
            except Exception:
                continue
            if name and price > 0:
                out.append({"name": name, "price": price})
    return out


async def _parse_pricelist_file(file_b64: str, mime: str, item_names: list):
    raw = (file_b64 or "").strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    if "pdf" in (mime or "").lower():
        imgs = _pdf_to_images_b64(base64.b64decode(raw))
    else:
        imgs = [raw]
    from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
    prompt = ("Extraia a tabela/lista de preços deste fornecedor. Itens de interesse: "
              + ", ".join(item_names) + ". Responda SOMENTE JSON: "
              '{"items":[{"name":"<produto>","price":<preco_unitario_numerico>}]} com o preço de cada item encontrado.')
    chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"quote-{uuid.uuid4().hex[:6]}",
                   system_message="Você extrai preços de tabelas/listas de fornecedores em português.").with_model("gemini", "gemini-3.1-pro-preview")
    resp = await asyncio.wait_for(chat.send_message(UserMessage(
        text=prompt, file_contents=[ImageContent(image_base64=b) for b in imgs])), timeout=120)
    text = (resp if isinstance(resp, str) else str(resp)).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    items = parsed.get("items", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
    return [{"name": i.get("name", ""), "price": float(i.get("price") or 0)} for i in items if i.get("name")]


@api_router.post("/quotes/{qid}/response")
async def quote_response(qid: str, data: QuoteResponseIn, user: dict = Depends(require_permission("products"))):
    q = await db.quotes.find_one({"_id": ObjectId(qid)})
    if not q:
        raise HTTPException(status_code=404, detail="Cotação não encontrada")
    item_names = [i.get("name", "") for i in q.get("items", [])]
    offers = []
    if data.file_base64:
        try:
            offers = await _parse_pricelist_file(data.file_base64, data.mime_type or "image/jpeg", item_names)
        except Exception as e:
            logger.warning(f"quote file parse failed: {e}")
            raise HTTPException(status_code=400, detail="Não foi possível ler o arquivo da tabela.")
    elif data.text:
        offers = _parse_price_text(data.text)
    if not offers:
        raise HTTPException(status_code=400, detail="Nenhum preço identificado na resposta.")
    responses = q.get("responses", {})
    responses[data.supplier_id] = offers
    await db.quotes.update_one({"_id": q["_id"]}, {"$set": {"responses": responses}})
    return {"supplier_id": data.supplier_id, "offers": offers, "count": len(offers)}


@api_router.post("/quotes/{qid}/simulate")
async def quote_simulate(qid: str, user: dict = Depends(require_permission("products"))):
    q = await db.quotes.find_one({"_id": ObjectId(qid)})
    if not q:
        raise HTTPException(status_code=404, detail="Cotação não encontrada")
    responses = q.get("responses", {})
    for sid in q.get("supplier_ids", []):
        responses[sid] = [{"name": i["name"], "price": round(_random.uniform(5, 60) * (1 + _random.uniform(-0.15, 0.15)), 2)}
                          for i in q.get("items", [])]
    await db.quotes.update_one({"_id": q["_id"]}, {"$set": {"responses": responses}})
    return {"ok": True, "suppliers": len(responses)}


@api_router.get("/quotes/{qid}/report")
async def quote_report(qid: str, user: dict = Depends(get_current_user)):
    q = await db.quotes.find_one({"_id": ObjectId(qid)})
    if not q:
        raise HTTPException(status_code=404, detail="Cotação não encontrada")
    sup_docs = await db.suppliers.find().to_list(1000)
    smap = {str(s["_id"]): s for s in sup_docs}
    responses = q.get("responses", {})
    blocks = {}
    unmatched = []
    total_savings = 0.0
    total_cost = 0.0
    for it in q.get("items", []):
        it_tokens = _tok(it["name"])
        it_norm = _norm(it["name"])
        offers = []
        for sid, offlist in responses.items():
            best, best_score = None, 0
            for o in offlist:
                on = o.get("name", "")
                score = len(it_tokens & _tok(on))
                if it_norm and (it_norm in _norm(on) or _norm(on) in it_norm):
                    score += 2
                if score > best_score:
                    best, best_score = o, score
            if best and best_score >= 1:
                offers.append((sid, float(best["price"]), best.get("name", "")))
        if not offers:
            unmatched.append(it["name"])
            continue
        offers.sort(key=lambda x: x[1])
        win_sid, win_price, win_name = offers[0]
        second = offers[1][1] if len(offers) > 1 else None
        qty = it.get("quantity", 1)
        savings = round((second - win_price) * qty, 2) if second is not None else 0.0
        total_savings += savings
        total_cost += round(win_price * qty, 2)
        b = blocks.setdefault(win_sid, {"supplier_id": win_sid,
                                        "supplier_name": smap.get(win_sid, {}).get("name", "Fornecedor"),
                                        "contact_name": smap.get(win_sid, {}).get("contact_name", ""),
                                        "whatsapp": smap.get(win_sid, {}).get("whatsapp", ""),
                                        "items": [], "subtotal": 0.0})
        b["items"].append({"name": it["name"], "quantity": qty, "unit_price": win_price,
                           "second_best": second, "savings": savings, "offer_name": win_name,
                           "offers_count": len(offers)})
        b["subtotal"] = round(b["subtotal"] + win_price * qty, 2)
    return {"title": q.get("title"), "blocks": list(blocks.values()), "unmatched": unmatched,
            "total_savings": round(total_savings, 2), "total_cost": round(total_cost, 2),
            "suppliers_responded": len(responses)}


# ------------------------------------------------------------------ Integrations (iFood / 99Food credentials)
class IfoodCredsIn(BaseModel):
    client_id: str
    client_secret: str
    merchant_id: str


class NineNineCredsIn(BaseModel):
    store_id: str
    api_secret: str


async def _tenant_slug(user: dict) -> str:
    t = user.get("_tenant") or {}
    if t.get("is_primary"):
        return "matriz"
    if t.get("_id"):
        doc = await cdb.tenants.find_one({"_id": t["_id"]})
        if doc and doc.get("slug"):
            return doc["slug"]
    return t.get("slug") or "matriz"


def _mask(s: str) -> str:
    if not s:
        return ""
    return (s[:4] + "••••" + s[-3:]) if len(s) > 9 else "••••"


@api_router.get("/integrations")
async def get_integrations(user: dict = Depends(require_permission("settings"))):
    s = await db.settings.find_one({"_id": "general"}) or {}
    integ = s.get("integrations") or {}
    ife = integ.get("ifood") or {}
    nin = integ.get("99food") or {}
    slug = await _tenant_slug(user)
    return {
        "ifood": {"connected": bool(ife.get("connected")), "merchant_id": ife.get("merchant_id", ""),
                  "client_id_masked": _mask(ife.get("client_id", "")), "connected_at": ife.get("connected_at")},
        "99food": {"connected": bool(nin.get("connected")), "store_id": nin.get("store_id", ""),
                   "api_secret_masked": _mask(nin.get("api_secret", "")), "connected_at": nin.get("connected_at")},
        "slug": slug,
        "webhook_ifood_path": f"/api/webhooks/ifood/{slug}",
        "webhook_99food_path": f"/api/webhooks/99food/{slug}",
    }


@api_router.put("/integrations/ifood")
async def connect_ifood(data: IfoodCredsIn, user: dict = Depends(require_permission("settings"))):
    if not (data.client_id.strip() and data.client_secret.strip() and data.merchant_id.strip()):
        raise HTTPException(status_code=400, detail="Preencha Client ID, Client Secret e Merchant ID")
    s = await db.settings.find_one({"_id": "general"}) or {}
    integ = s.get("integrations") or {}
    integ["ifood"] = {"client_id": data.client_id.strip(), "client_secret": data.client_secret.strip(),
                      "merchant_id": data.merchant_id.strip(), "connected": True, "connected_at": now_iso()}
    await db.settings.update_one({"_id": "general"}, {"$set": {"integrations": integ}}, upsert=True)
    return {"ok": True, "connected": True, "merchant_id": data.merchant_id.strip()}


@api_router.put("/integrations/99food")
async def connect_99food(data: NineNineCredsIn, user: dict = Depends(require_permission("settings"))):
    if not (data.store_id.strip() and data.api_secret.strip()):
        raise HTTPException(status_code=400, detail="Preencha Store ID e API Secret Key")
    s = await db.settings.find_one({"_id": "general"}) or {}
    integ = s.get("integrations") or {}
    integ["99food"] = {"store_id": data.store_id.strip(), "api_secret": data.api_secret.strip(),
                       "connected": True, "connected_at": now_iso()}
    await db.settings.update_one({"_id": "general"}, {"$set": {"integrations": integ}}, upsert=True)
    slug = await _tenant_slug(user)
    return {"ok": True, "connected": True, "webhook_99food_path": f"/api/webhooks/99food/{slug}"}


@api_router.post("/integrations/{provider}/disconnect")
async def disconnect_integration(provider: str, user: dict = Depends(require_permission("settings"))):
    key = "ifood" if provider == "ifood" else "99food"
    s = await db.settings.find_one({"_id": "general"}) or {}
    integ = s.get("integrations") or {}
    cur = integ.get(key) or {}
    cur["connected"] = False
    integ[key] = cur
    await db.settings.update_one({"_id": "general"}, {"$set": {"integrations": integ}}, upsert=True)
    return {"ok": True, "connected": False}


# ------------------------------------------------------------------ Object Storage (attachments)
STORAGE_BASE = (os.environ.get("INTEGRATION_PROXY_URL") or "").strip() or "https://integrations.emergentagent.com"
STORAGE_URL = STORAGE_BASE.rstrip("/") + "/objstore/api/v1/storage"
APP_NAME = "superionpro"
_storage_key = None


def init_storage(force: bool = False):
    global _storage_key
    if _storage_key and not force:
        return _storage_key
    resp = requests.post(f"{STORAGE_URL}/init", json={"emergent_key": EMERGENT_LLM_KEY}, timeout=30)
    resp.raise_for_status()
    _storage_key = resp.json()["storage_key"]
    return _storage_key


def put_object(path: str, data: bytes, content_type: str) -> dict:
    key = init_storage()
    resp = requests.put(f"{STORAGE_URL}/objects/{path}",
                        headers={"X-Storage-Key": key, "Content-Type": content_type}, data=data, timeout=120)
    if resp.status_code == 404:
        key = init_storage(force=True)
        resp = requests.put(f"{STORAGE_URL}/objects/{path}",
                            headers={"X-Storage-Key": key, "Content-Type": content_type}, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_object(path: str):
    key = init_storage()
    resp = requests.get(f"{STORAGE_URL}/objects/{path}", headers={"X-Storage-Key": key}, timeout=60)
    if resp.status_code == 404:
        key = init_storage(force=True)
        resp = requests.get(f"{STORAGE_URL}/objects/{path}", headers={"X-Storage-Key": key}, timeout=60)
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "application/octet-stream")


MIME_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp",
            "application/pdf": "pdf", "image/gif": "gif"}


@api_router.post("/uploads")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    data = await file.read()
    ct = file.content_type or "application/octet-stream"
    ext = MIME_EXT.get(ct, (file.filename.split(".")[-1] if "." in (file.filename or "") else "bin"))
    path = f"{APP_NAME}/uploads/{user['_id']}/{uuid.uuid4().hex}.{ext}"
    try:
        result = await asyncio.to_thread(put_object, path, data, ct)
    except Exception as e:
        logger.warning(f"upload failed: {e}")
        raise HTTPException(status_code=502, detail="Falha ao enviar arquivo para o armazenamento")
    await cdb.files.insert_one({"storage_path": result["path"], "original_filename": file.filename or "arquivo",
                                "content_type": ct, "size": result.get("size", len(data)), "is_deleted": False,
                                "owner": str(user["_id"]), "tenant_id": str((user.get("_tenant") or {}).get("_id", "")),
                                "created_at": now_iso()})
    return {"path": result["path"], "url": f"/api/uploads/{result['path']}", "content_type": ct,
            "name": file.filename or "arquivo"}


@api_router.get("/uploads/{path:path}")
async def download_file(path: str, user: dict = Depends(get_current_user)):
    rec = await cdb.files.find_one({"storage_path": path, "is_deleted": False})
    if not rec:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    tid = str((user.get("_tenant") or {}).get("_id", ""))
    if not (rec.get("tenant_id") == tid or rec.get("owner") == str(user["_id"])):
        raise HTTPException(status_code=403, detail="Acesso negado")
    try:
        content, ct = await asyncio.to_thread(get_object, path)
    except Exception:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return Response(content=content, media_type=rec.get("content_type") or ct)


# ------------------------------------------------------------------ Customers directory
class CustomerIn(BaseModel):
    name: str
    phone: str = ""
    whatsapp: str = ""
    segment: str = ""
    notes: str = ""


def _customer_out(c):
    c["id"] = str(c.pop("_id"))
    return c


@api_router.get("/customers")
async def list_customers(user: dict = Depends(get_current_user)):
    out = await db.customers.find().sort("name", 1).to_list(10000)
    return [_customer_out(c) for c in out]


@api_router.get("/customers/segments")
async def customer_segments(user: dict = Depends(get_current_user)):
    segs = await db.customers.distinct("segment")
    return sorted([s for s in segs if s])


@api_router.post("/customers")
async def create_customer(data: CustomerIn, user: dict = Depends(require_permission("pos"))):
    doc = data.model_dump()
    doc["created_at"] = now_iso()
    res = await db.customers.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    return doc


@api_router.put("/customers/{cid}")
async def update_customer(cid: str, data: CustomerIn, user: dict = Depends(require_permission("pos"))):
    await db.customers.update_one({"_id": ObjectId(cid)}, {"$set": data.model_dump()})
    return _customer_out(await db.customers.find_one({"_id": ObjectId(cid)}))


@api_router.delete("/customers/{cid}")
async def delete_customer(cid: str, user: dict = Depends(require_permission("pos"))):
    await db.customers.delete_one({"_id": ObjectId(cid)})
    return {"ok": True}


@api_router.post("/customers/sync-from-orders")
async def sync_customers_from_orders(user: dict = Depends(require_permission("pos"))):
    existing = {re.sub(r"\D", "", (c.get("whatsapp") or c.get("phone") or ""))
                for c in await db.customers.find().to_list(20000)}
    seen = set(x for x in existing if x)
    added = 0
    for o in await db.orders.find().sort("created_at", -1).to_list(5000):
        cust = o.get("customer") or {}
        phone_raw = cust.get("phone") or ""
        phone = re.sub(r"\D", "", phone_raw)
        if phone and phone not in seen:
            seen.add(phone)
            await db.customers.insert_one({"name": cust.get("name") or "Cliente", "phone": phone_raw,
                                           "whatsapp": phone_raw, "segment": "pedidos",
                                           "notes": "Importado de pedidos", "created_at": now_iso()})
            added += 1
    return {"added": added}


# ------------------------------------------------------------------ Bulk Messaging (approval-gated queue)
WA_SEND_DELAY = float(os.environ.get("WA_SEND_DELAY", "1.0"))


class Attachment(BaseModel):
    path: str = ""
    url: str = ""
    name: str = ""
    content_type: str = ""


class CampaignIn(BaseModel):
    title: str = "Campanha"
    audience: str = "customers"       # customers | suppliers | segment
    segment: str = ""
    recipient_ids: List[str] = []
    body: str = ""
    attachments: List[Attachment] = []
    scheduled_at: Optional[str] = None
    source: str = "manual"            # manual | ai


async def _company_name():
    s = await db.settings.find_one({"_id": "general"}) or {}
    return s.get("companyName") or s.get("tradeName") or "Nossa Loja"


def _render_tpl(body: str, name: str, store: str) -> str:
    return ((body or "").replace("{{nome_cliente}}", name or "")
            .replace("{{nome_fornecedor}}", name or "")
            .replace("{{nome_loja}}", store or ""))


async def _resolve_recipients(audience: str, segment: str = "", recipient_ids=None):
    recipient_ids = recipient_ids or []
    out = []
    if audience == "suppliers":
        for s in await db.suppliers.find().sort("name", 1).to_list(5000):
            if recipient_ids and str(s["_id"]) not in recipient_ids:
                continue
            phone = s.get("whatsapp", "")
            if phone:
                out.append({"id": str(s["_id"]), "name": s.get("name", ""), "phone": phone, "type": "supplier"})
    else:
        q = {"segment": segment} if (audience == "segment" and segment) else {}
        for c in await db.customers.find(q).sort("name", 1).to_list(20000):
            if recipient_ids and str(c["_id"]) not in recipient_ids:
                continue
            phone = c.get("whatsapp") or c.get("phone") or ""
            if phone:
                out.append({"id": str(c["_id"]), "name": c.get("name", ""), "phone": phone, "type": "customer"})
    return out


def _campaign_out(c):
    c["id"] = str(c.pop("_id"))
    return c


@api_router.get("/v1/messaging/campaigns")
async def list_campaigns(user: dict = Depends(get_current_user)):
    return [_campaign_out(c) for c in await db.campaigns.find().sort("created_at", -1).to_list(500)]


@api_router.get("/v1/messaging/recipients")
async def preview_recipients(audience: str = "customers", segment: str = "",
                             user: dict = Depends(get_current_user)):
    recs = await _resolve_recipients(audience, segment)
    return {"count": len(recs), "recipients": recs[:300]}


@api_router.post("/v1/messaging/campaigns")
async def create_campaign(data: CampaignIn, user: dict = Depends(require_permission("pos"))):
    recs = await _resolve_recipients(data.audience, data.segment, data.recipient_ids)
    doc = {"title": data.title, "audience": data.audience, "segment": data.segment,
           "recipient_ids": data.recipient_ids, "body": data.body,
           "attachments": [a.model_dump() for a in data.attachments],
           "scheduled_at": data.scheduled_at, "source": data.source, "status": "draft",
           "recipient_count": len(recs), "counts": {"queued": 0, "sent": 0, "delivered": 0, "read": 0, "failed": 0},
           "created_at": now_iso(), "created_by": user.get("name")}
    res = await db.campaigns.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    doc.pop("_id", None)
    doc["preview_recipients"] = recs[:5]
    return doc


@api_router.delete("/v1/messaging/campaigns/{cid}")
async def delete_campaign(cid: str, user: dict = Depends(require_permission("pos"))):
    await db.campaigns.delete_one({"_id": ObjectId(cid)})
    await db.messages.delete_many({"campaign_id": cid})
    return {"ok": True}


async def _approve_campaign(cid: str, actor: dict):
    """Core approval: expands recipients, enqueues per-recipient messages, starts sender.
    This is the ONLY path that actually dispatches; always triggered by an explicit user action."""
    # Atomic draft->sending transition prevents duplicate enqueue on rapid double-approve
    c = await db.campaigns.find_one_and_update(
        {"_id": ObjectId(cid), "status": "draft"}, {"$set": {"status": "sending", "approved_at": now_iso()}})
    if not c:
        existing = await db.campaigns.find_one({"_id": ObjectId(cid)})
        if not existing:
            raise HTTPException(status_code=404, detail="Campanha não encontrada")
        raise HTTPException(status_code=400, detail="Esta campanha já foi aprovada/enviada")
    recs = await _resolve_recipients(c.get("audience"), c.get("segment", ""), c.get("recipient_ids", []))
    if not recs:
        await db.campaigns.update_one({"_id": c["_id"]}, {"$set": {"status": "draft"}})
        raise HTTPException(status_code=400, detail="Nenhum destinatário com WhatsApp válido para esta campanha")
    store = await _company_name()
    now = now_iso()
    msg_docs = [{"campaign_id": cid, "to_name": r["name"], "to_phone": r["phone"], "to_type": r["type"],
                 "body": _render_tpl(c.get("body", ""), r["name"], store), "attachments": c.get("attachments", []),
                 "status": "queued", "error": "", "created_at": now,
                 "sent_at": None, "delivered_at": None, "read_at": None} for r in recs]
    await db.messages.insert_many(msg_docs)
    await db.campaigns.update_one({"_id": c["_id"]}, {"$set": {
        "recipient_count": len(recs),
        "counts": {"queued": len(recs), "sent": 0, "delivered": 0, "read": 0, "failed": 0}}})
    await write_audit(actor, "CREATE", "Disparos", cid, new_values={
        "titulo": c.get("title"), "publico": c.get("audience"), "destinatarios": len(recs),
        "origem": c.get("source")}, resource_name=c.get("title", ""))
    asyncio.create_task(_run_sender(cid))
    return {"ok": True, "status": "sending", "queued": len(recs)}


@api_router.post("/v1/messaging/campaigns/{cid}/approve")
async def approve_campaign(cid: str, user: dict = Depends(require_permission("pos"))):
    return await _approve_campaign(cid, user)


async def _run_sender(cid: str):
    """Simulated rate-limited WhatsApp dispatcher. Advances per-message status over time."""
    try:
        msgs = await db.messages.find({"campaign_id": cid, "status": "queued"}).to_list(20000)
        for m in msgs:
            try:
                await asyncio.sleep(WA_SEND_DELAY)
                if _random.random() < 0.06:
                    await db.messages.update_one({"_id": m["_id"]}, {"$set": {
                        "status": "failed", "error": "Número inválido ou sem WhatsApp", "sent_at": now_iso()}})
                    await db.campaigns.update_one({"_id": ObjectId(cid)}, {"$inc": {"counts.queued": -1, "counts.failed": 1}})
                    continue
                await db.messages.update_one({"_id": m["_id"]}, {"$set": {"status": "sent", "sent_at": now_iso()}})
                await db.campaigns.update_one({"_id": ObjectId(cid)}, {"$inc": {"counts.queued": -1, "counts.sent": 1}})
                await asyncio.sleep(0.4)
                await db.messages.update_one({"_id": m["_id"]}, {"$set": {"status": "delivered", "delivered_at": now_iso()}})
                await db.campaigns.update_one({"_id": ObjectId(cid)}, {"$inc": {"counts.sent": -1, "counts.delivered": 1}})
                if _random.random() < 0.7:
                    await asyncio.sleep(0.5)
                    await db.messages.update_one({"_id": m["_id"]}, {"$set": {"status": "read", "read_at": now_iso()}})
                    await db.campaigns.update_one({"_id": ObjectId(cid)}, {"$inc": {"counts.delivered": -1, "counts.read": 1}})
            except Exception as me:
                logger.warning(f"sender message {m.get('_id')} failed: {me}")
        await db.campaigns.update_one({"_id": ObjectId(cid)}, {"$set": {"status": "completed", "completed_at": now_iso()}})
    except Exception as e:
        logger.warning(f"sender failed for {cid}: {e}")


@api_router.get("/v1/messaging/queue")
async def messaging_queue(campaign_id: str = "", user: dict = Depends(get_current_user)):
    q = {"campaign_id": campaign_id} if campaign_id else {}
    msgs = await db.messages.find(q).sort("created_at", -1).to_list(3000)
    return [{"id": str(m["_id"]), "campaign_id": m.get("campaign_id"), "to_name": m.get("to_name"),
             "to_phone": m.get("to_phone"), "to_type": m.get("to_type"), "body": m.get("body"),
             "status": m.get("status"), "error": m.get("error", ""), "sent_at": m.get("sent_at"),
             "delivered_at": m.get("delivered_at"), "read_at": m.get("read_at"),
             "attachments": m.get("attachments", [])} for m in msgs]


# ------------------------------------------------------------------ AI: multimodal order parser
PARSE_ORDER_SYSTEM = """Você extrai itens de um pedido de compra a partir de texto ou imagem
(lista manuscrita, impressa ou print de conversa). Responda SOMENTE JSON válido (sem markdown):
{"items":[{"name":"<produto>","quantity":<numero>,"unit":"<unidade: caixa, kg, un, fardo, pacote, litro>"}]}
Interprete expressões como "3 caixas de óleo" (quantity 3, unit caixa, name óleo),
"50kg de batata" (quantity 50, unit kg, name batata), "2 fardos de refrigerante".
Use português do Brasil. Se a quantidade não estiver clara, use 1."""


def _best_product_match(name: str, products: list):
    nt = _tok(name)
    nn = _norm(name)
    best, score = None, 0
    for p in products:
        pn = p.get("name", "")
        s = len(nt & _tok(pn))
        if nn and (nn in _norm(pn) or _norm(pn) in nn):
            s += 2
        if s > score:
            best, score = p, s
    return best if score >= 1 else None


class ParseOrderIn(BaseModel):
    text: Optional[str] = None
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None


@api_router.post("/v1/ai/parse-order-input")
async def parse_order_input(data: ParseOrderIn, user: dict = Depends(get_current_user),
                            _gate: dict = Depends(require_active_tenant)):
    items = []
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"order-{user['_id']}-{uuid.uuid4().hex[:6]}",
                       system_message=PARSE_ORDER_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        if data.file_base64:
            raw = data.file_base64.split(",", 1)[1] if data.file_base64.startswith("data:") else data.file_base64
            imgs = _pdf_to_images_b64(base64.b64decode(raw)) if "pdf" in (data.mime_type or "").lower() else [raw]
            resp = await asyncio.wait_for(chat.send_message(UserMessage(
                text="Extraia todos os itens do pedido nesta imagem no formato JSON solicitado.",
                file_contents=[ImageContent(image_base64=b) for b in imgs])), timeout=120)
        elif (data.text or "").strip():
            resp = await asyncio.wait_for(chat.send_message(UserMessage(
                text=f"Extraia os itens do pedido a seguir:\n{data.text}")), timeout=90)
        else:
            raise HTTPException(status_code=400, detail="Envie um texto ou uma imagem do pedido")
        txt = (resp if isinstance(resp, str) else str(resp)).strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            if txt.startswith("json"):
                txt = txt[4:]
        parsed = json.loads(txt.strip())
        items = parsed.get("items", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"parse-order failed: {e}")
        raise HTTPException(status_code=400, detail="Não foi possível interpretar o pedido. Tente novamente.")
    products = await db.products.find().to_list(3000)
    out = []
    for it in items[:60]:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        match = _best_product_match(name, products)
        out.append({"raw_name": name, "quantity": it.get("quantity") or 1, "unit": it.get("unit") or "un",
                    "matched": bool(match), "product_id": str(match["_id"]) if match else None,
                    "matched_name": match.get("name") if match else "",
                    "price": match.get("price", 0) if match else 0,
                    "cost": match.get("cost", 0) if match else 0})
    return {"count": len(out), "matched": sum(1 for o in out if o["matched"]), "items": out}


# ------------------------------------------------------------------ AI: system-wide agent (tool-use + approval guardrail)
AGENT_SYSTEM = """Você é o Assistente IA do Superion Pro (ERP/PDV, português do Brasil).
REGRA DE SEGURANÇA CRÍTICA E INEGOCIÁVEL: você NUNCA envia mensagens de WhatsApp, pedidos a fornecedores
ou disparos a clientes por conta própria. Ao preparar um pedido ou disparo, use APENAS a ferramenta de rascunho
correspondente — ela só cria um RASCUNHO que o usuário precisa aprovar manualmente na interface.

Responda SEMPRE com um ÚNICO JSON válido, sem markdown.
Para chamar uma ferramenta: {"tool":"<nome>","args":{...}}
Para finalizar: {"final":{...}}

Ferramentas de LEITURA (use quando precisar de dados reais):
- get_inventory_status  args: {"low_only": true|false}
- get_sales_summary     args: {"period":"today"|"month"}
- get_supplier_prices   args: {"product":"<opcional>"}

Ferramentas de AÇÃO (apenas criam RASCUNHO, nunca enviam):
- prepare_supplier_whatsapp_order  args: {"supplier_id":"<id>","items":[{"name":..,"quantity":..}]}
- prepare_broadcast                args: {"audience":"customers"|"suppliers","segment":"","title":"...","body":"<texto; pode usar {{nome_cliente}} e {{nome_loja}}>"}
- update_product                   args: {"query":"<nome ou EAN>","fields":{<campos: name,ean,brand,category,cost,margin,price,stock_store,stock_deposit,min_stock>}}

Ferramentas de CATÁLOGO (aplicam a mudança na hora e confirmam):
- disable_catalog_product args: {"target":"<nome/sku>"}
- enable_catalog_product  args: {"target":"<nome/sku>"}
- update_catalog_price    args: {"target":"<nome/sku>","new_price":<numero>}
- bulk_toggle_category    args: {"category":"<categoria>","status":"active"|"paused"}

Ferramentas de CATÁLOGO (aplicam a mudança na hora e confirmam):
- disable_catalog_product args: {"target":"<nome/sku>"}   (oculta item do catálogo público)
- enable_catalog_product  args: {"target":"<nome/sku>"}   (reativa item no catálogo)
- update_catalog_price    args: {"target":"<nome/sku>","new_price":<numero>}
- bulk_toggle_category    args: {"category":"<categoria>","status":"active"|"paused"}

Ações de INTERFACE:
- navigate  args: {"value":"dashboard|pos|orders|products|suppliers|purchasing|reports|customers|messaging|settings"}
- set_theme args: {"value":"light"|"dark"}

Formatos de final possíveis:
- {"type":"answer","reply":"<resposta curta em pt-BR>"}
- {"type":"ui_action","action":"navigate","value":"...","reply":"..."}
- {"type":"ui_action","action":"set_theme","value":"...","reply":"..."}

Quando criar um rascunho de pedido/disparo, finalize com um "answer" explicando que preparou o rascunho
e pedindo para o usuário revisar e clicar em "Aprovar e Enviar Agora". Seja conciso."""

APPROVAL_RE = re.compile(r"(pode enviar|envie agora|enviar agora|aprovar|aprovo|aprovado|confirmar envio|"
                         r"pode mandar|manda agora|disparar agora|pode disparar)", re.I)


def _count_by(rows, key):
    out = {}
    for r in rows:
        out[r.get(key, "?")] = out.get(r.get(key, "?"), 0) + 1
    return out


async def _prepare_supplier_order(sid, items, user):
    if not sid or not ObjectId.is_valid(str(sid)):
        return {"error": "supplier_id inválido"}
    s = await db.suppliers.find_one({"_id": ObjectId(sid)})
    if not s:
        return {"error": "fornecedor não encontrado"}
    items = [it for it in (items or []) if it.get("name")]
    lines = "\n".join([f"- {it.get('quantity', 1)}x {it.get('name', '')}" for it in items])
    store = await _company_name()
    body = (f"Olá {s.get('contact_name') or s.get('name')}! Gostaríamos de fazer o seguinte pedido:\n\n"
            f"{lines}\n\nPode confirmar disponibilidade e valores? Obrigado!\n{store}")
    doc = {"title": f"Pedido — {s.get('name')}", "audience": "suppliers", "segment": "",
           "recipient_ids": [str(sid)], "body": body, "attachments": [], "scheduled_at": None,
           "source": "ai", "status": "draft", "recipient_count": 1, "order_items": items,
           "counts": {"queued": 0, "sent": 0, "delivered": 0, "read": 0, "failed": 0},
           "created_at": now_iso(), "created_by": user.get("name")}
    res = await db.campaigns.insert_one(doc)
    cid = str(res.inserted_id)
    return {"_draft_campaign": {"id": cid, "kind": "supplier_order", "title": doc["title"],
                                "supplier_name": s.get("name"), "to_phone": s.get("whatsapp", ""),
                                "body": body, "items": items}, "created_draft_id": cid}


async def _prepare_broadcast(args, user):
    audience = args.get("audience", "customers")
    segment = args.get("segment", "") or ""
    body = args.get("body", "") or ""
    title = args.get("title", "Disparo IA")
    recs = await _resolve_recipients(audience, segment)
    doc = {"title": title, "audience": audience, "segment": segment, "recipient_ids": [],
           "body": body, "attachments": [], "scheduled_at": None, "source": "ai", "status": "draft",
           "recipient_count": len(recs), "counts": {"queued": 0, "sent": 0, "delivered": 0, "read": 0, "failed": 0},
           "created_at": now_iso(), "created_by": user.get("name")}
    res = await db.campaigns.insert_one(doc)
    cid = str(res.inserted_id)
    return {"_draft_campaign": {"id": cid, "kind": "broadcast", "title": title, "audience": audience,
                                "segment": segment, "body": body, "recipient_count": len(recs),
                                "sample": [r["name"] for r in recs[:5]]}, "created_draft_id": cid}


async def _agent_update_product(value, user):
    if not has_permission(user, "products"):
        return {"error": "sem permissão para editar produtos"}
    q = (value or {}).get("query", "")
    fields = (value or {}).get("fields", {}) or {}
    prod = None
    if q:
        prod = await db.products.find_one({"ean": str(q).strip()})
        if not prod:
            prod = await db.products.find_one({"name": {"$regex": re.escape(str(q)), "$options": "i"}})
    if not prod:
        return {"error": f"produto '{q}' não encontrado"}
    allowed = {"name", "ean", "brand", "category", "cost", "margin", "price",
               "stock_store", "stock_deposit", "min_stock"}
    num_fields = {"cost", "margin", "price", "stock_store", "stock_deposit", "min_stock"}
    upd = {}
    for k, v in fields.items():
        if k not in allowed or v is None:
            continue
        upd[k] = float(v) if k in num_fields else str(v)
    if not upd:
        return {"error": "nenhum campo válido para atualizar"}
    merged = {**prod, **upd}
    upd["price"] = compute_price(merged.get("cost", 0), merged.get("cost_qty", 1), merged.get("margin", 30),
                                 upd.get("price", prod.get("price")) if "price" in fields else None)
    await db.products.update_one({"_id": prod["_id"]}, {"$set": upd})
    return {"ok": True, "product": prod.get("name"), "updated": list(fields.keys())}


async def _run_agent_tool(name, args, user):
    args = args or {}
    if name == "get_inventory_status":
        low_only = bool(args.get("low_only"))
        prods = await db.products.find().to_list(3000)
        rows = []
        for p in prods:
            st = p.get("stock_store", 0) + p.get("stock_deposit", 0)
            low = (0 < st <= p.get("min_stock", 5)) or st <= 0
            if low_only and not low:
                continue
            rows.append({"name": p.get("name"), "stock": round(st, 2), "min": p.get("min_stock", 5),
                         "low": low, "price": p.get("price", 0)})
        return {"total_products": len(prods), "items": rows[:80],
                "low_count": sum(1 for p in prods if 0 < (p.get("stock_store", 0) + p.get("stock_deposit", 0)) <= p.get("min_stock", 5)),
                "out_of_stock": sum(1 for p in prods if (p.get("stock_store", 0) + p.get("stock_deposit", 0)) <= 0)}
    if name == "get_sales_summary":
        period = args.get("period", "today")
        sales = await db.sales.find().to_list(5000)
        orders = await db.orders.find().to_list(5000)
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        month = today[:7]

        def inp(ci):
            d = (ci or "")[:10]
            return (d[:7] == month) if period == "month" else (d == today)
        s_sel = [s for s in sales if inp(s.get("created_at"))]
        o_sel = [o for o in orders if inp(o.get("created_at"))]
        return {"period": period, "pos_revenue": round(sum(s.get("total", 0) for s in s_sel), 2),
                "pos_count": len(s_sel), "pos_profit": round(sum(s.get("profit", 0) for s in s_sel), 2),
                "orders_count": len(o_sel), "orders_total": round(sum(o.get("total", 0) for o in o_sel), 2),
                "orders_by_source": _count_by(o_sel, "source")}
    if name == "get_supplier_prices":
        product = (args.get("product") or "").strip().lower()
        quotes = await db.quotes.find().sort("created_at", -1).to_list(200)
        suppliers = {str(s["_id"]): s for s in await db.suppliers.find().to_list(1000)}
        latest = {}
        for q in quotes:
            for sid, offers in (q.get("responses") or {}).items():
                sname = suppliers.get(sid, {}).get("name", "Fornecedor")
                for o in offers:
                    on = o.get("name", "")
                    if product and product not in on.lower():
                        continue
                    key = (sname, on.lower())
                    if key not in latest:
                        latest[key] = {"supplier": sname, "supplier_id": sid, "product": on, "price": o.get("price")}
        return {"prices": list(latest.values())[:80],
                "suppliers": [{"id": sid, "name": s.get("name"), "whatsapp": s.get("whatsapp", "")}
                              for sid, s in suppliers.items()]}
    if name == "prepare_supplier_whatsapp_order":
        if not has_permission(user, "pos"):
            return {"error": "Sem permissão para preparar pedidos a fornecedores."}
        return await _prepare_supplier_order(args.get("supplier_id"), args.get("items", []), user)
    if name == "prepare_broadcast":
        if not has_permission(user, "pos"):
            return {"error": "Sem permissão para preparar disparos."}
        return await _prepare_broadcast(args, user)
    if name == "update_product":
        return await _agent_update_product(args, user)
    if name in ("disable_catalog_product", "enable_catalog_product", "update_catalog_price", "bulk_toggle_category"):
        if not has_permission(user, "products"):
            return {"type": "answer", "reply": "Você não tem permissão para gerenciar o catálogo."}
        res = await _catalog_execute(name, args.get("target") or args.get("product_name_or_sku", ""),
                                     args.get("new_price") if "new_price" in args else args.get("value"),
                                     args.get("category_name") or args.get("category", ""), args.get("status"), user)
        return {"type": "answer", "reply": res["message"]}
    return {"error": f"ferramenta desconhecida: {name}"}


class AgentIn(BaseModel):
    message: str
    pending_draft_id: Optional[str] = None


@api_router.post("/v1/ai/agent")
async def ai_agent(data: AgentIn, user: dict = Depends(get_current_user)):
    # Explicit typed approval of a pending draft (guardrail-compliant command path)
    if data.pending_draft_id and APPROVAL_RE.search(data.message or ""):
        if not has_permission(user, "pos"):
            return {"type": "answer", "reply": "Você não tem permissão para aprovar/enviar disparos."}
        try:
            res = await _approve_campaign(data.pending_draft_id, user)
            return {"type": "approved", "campaign_id": data.pending_draft_id,
                    "reply": f"Perfeito! Aprovado e enviado para a fila ({res.get('queued', 0)} destinatário(s)). "
                             "Acompanhe o status em Disparo em Massa."}
        except HTTPException as e:
            msg = e.detail if isinstance(e.detail, str) else "não foi possível enviar"
            return {"type": "answer", "reply": f"Não consegui enviar: {msg}"}
    draft = None
    final = None
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"agent-{user['_id']}-{uuid.uuid4().hex[:6]}",
                       system_message=AGENT_SYSTEM).with_model("gemini", "gemini-3.1-pro-preview")
        msg = UserMessage(text=data.message)
        for _ in range(5):
            resp = await asyncio.wait_for(chat.send_message(msg), timeout=90)
            txt = (resp if isinstance(resp, str) else str(resp)).strip()
            if txt.startswith("```"):
                txt = txt.strip("`")
                if txt.startswith("json"):
                    txt = txt[4:]
            obj = json.loads(txt.strip())
            if isinstance(obj, dict) and "tool" in obj:
                result = await _run_agent_tool(obj.get("tool"), obj.get("args", {}), user)
                if isinstance(result, dict) and result.get("_draft_campaign"):
                    draft = result.pop("_draft_campaign")
                msg = UserMessage(text=f"RESULTADO da ferramenta {obj.get('tool')}: "
                                       f"{json.dumps(result, ensure_ascii=False)[:4000]}")
                continue
            final = obj.get("final", obj) if isinstance(obj, dict) else None
            break
    except Exception as e:
        logger.warning(f"agent failed: {e}")
        return {"type": "answer", "reply": "Desculpe, não consegui processar agora. Pode reformular?"}
    final = final or {"type": "answer", "reply": "Feito."}
    out = dict(final)
    if draft:
        out["type"] = "draft"
        out["draft"] = draft
        out.setdefault("reply", "Preparei um rascunho para você revisar. Confira e clique em "
                                "\"Aprovar e Enviar Agora\" para disparar.")
    return out



# ------------------------------------------------------------------ Seed + startup
async def seed():
    await db.users.create_index("username", unique=True)
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@superionpro.com")
    admin_username = os.environ.get("ADMIN_USERNAME", "alex borges").lower().strip()
    admin_name = os.environ.get("ADMIN_NAME", "Alex Borges")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Superion12#")
    # Remove stale default auto-seed admin when a custom admin username is configured
    if admin_username != "admin":
        await db.users.delete_one({"username": "admin", "role": "admin"})
    existing = await db.users.find_one({"username": admin_username})
    if not existing:
        await db.users.insert_one({"name": admin_name, "username": admin_username, "email": admin_email,
                                   "password_hash": hash_password(admin_password), "role": "admin",
                                   "permissions": ALL_PERMISSIONS, "active": True, "created_at": now_iso()})
        logger.info("Admin seeded")
    else:
        await db.users.update_one({"username": admin_username},
                                  {"$set": {"name": admin_name, "role": "admin", "permissions": ALL_PERMISSIONS,
                                            "active": True, "password_hash": hash_password(admin_password)}})
    admin = await cdb.users.find_one({"username": admin_username})
    primary = await cdb.tenants.find_one({"is_primary": True})
    if not primary:
        res = await cdb.tenants.insert_one({"company_name": f"{admin_name} (Matriz)", "is_primary": True,
                                            "status": "paid_active", "slug": "matriz", "created_at": now_iso()})
        primary = {"_id": res.inserted_id}
    if primary and not primary.get("slug"):
        await cdb.tenants.update_one({"_id": primary["_id"]}, {"$set": {"slug": "matriz"}})
    if admin and not admin.get("tenant_id"):
        await cdb.users.update_one({"_id": admin["_id"]}, {"$set": {"tenant_id": str(primary["_id"])}})
    await cdb.tenants.create_index("is_primary")
    try:
        await cdb.invites.create_index("token", unique=True)
    except Exception:
        pass
    if not await db.settings.find_one({"_id": "general"}):
        await db.settings.insert_one({"_id": "general", "defaultTheme": "light", "defaultMargin": 30.0,
                                      "commissionRate": 5.0, "companyName": "Superion Pro", "tradeName": "",
                                      "companyDoc": "", "companyPhone": "", "companyAddress": "", "logo": "",
                                      "receiptFooter": "Obrigado pela preferência! Volte sempre."})


@app.on_event("startup")
async def on_startup():
    await seed()
    try:
        await asyncio.to_thread(init_storage)
        logger.info("Object storage initialized")
    except Exception as e:
        logger.warning(f"storage init failed (will retry on first upload): {e}")


@api_router.get("/")
async def root():
    return {"message": "Superion Pro API online"}


app.include_router(api_router)

app.add_middleware(CORSMiddleware, allow_credentials=True,
                   allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
