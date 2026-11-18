"""Noeud Fog B -- Analyse Spatio-Temporelle, Architecture Maillée (PC 2).

NOUVELLE ARCHITECTURE (Star-Mesh) :
  Fog A -> Fog B : POST /analyze (transaction avec score IA)   [lien A<->B]
  Fog B -> Fog C : POST /submit_geo (geo direct, sans passer par A) [lien B<->C]
  Fog C -> Fog B : POST /receive_decision (broadcast final)    [lien C->B]

SECURITE :
  - HTTPS/TLS : canal chiffre, certificats signes par le CA interne
  - mTLS      : Fog B presente son certificat fog_b.crt
  - JWT HS256 : valide l'emetteur de chaque requete entrante

Lancer : py app.py [--port 8002]
"""
import argparse
import asyncio
import json
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from decision_engine import make_decision
from geo import CITIES, ATMS, get_coordinates, haversine_km
from inference import BehaviorScorer, fetch_history
from security import NodeAuth, build_client_ssl_ctx, require_from

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

NODE_ID = CONFIG["node_id"]
FOG_A_URL = CONFIG["peers"]["fog_a_url"]
FOG_C_URL = CONFIG["peers"]["fog_c_url"]
MAX_SPEED_KMH = CONFIG["thresholds"]["max_speed_kmh"]
MIN_DISTANCE_KM = CONFIG["thresholds"]["min_distance_km"]
SYNC_INTERVAL = CONFIG["resilience"]["sync_interval_s"]
PEER_TIMEOUT = CONFIG["resilience"]["peer_timeout_s"]

SEC = CONFIG["security"]
auth = NodeAuth(node_id=NODE_ID, secret=SEC["jwt_secret"])
_ssl_ctx = build_client_ssl_ctx(SEC["ca_cert"], SEC["node_cert"], SEC["node_key"])

logging.basicConfig(level=logging.INFO,
                    format="[FOG-B] %(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fog_b")

# ---------------------------------------------------------------------------
# Schemas SQLite
# ---------------------------------------------------------------------------
LOCATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS card_locations (
    card_id     TEXT PRIMARY KEY,
    last_atm_id TEXT,
    last_city   TEXT NOT NULL,
    last_lat    REAL NOT NULL,
    last_lon    REAL NOT NULL,
    last_ts     INTEGER NOT NULL
);
"""
TRANSACTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id TEXT NOT NULL, ts INTEGER NOT NULL, amount REAL NOT NULL
);
"""
OFFLINE_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS offline_queue (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id             TEXT NOT NULL,
    card_id           TEXT NOT NULL,
    amount            REAL NOT NULL,
    city              TEXT NOT NULL,
    timestamp         INTEGER NOT NULL,
    fog_a_score       REAL,
    impossible_travel INTEGER,
    decision          TEXT NOT NULL,
    reason            TEXT NOT NULL,
    decided_at        TEXT NOT NULL,
    synced            INTEGER NOT NULL DEFAULT 0
);
"""

# ---------------------------------------------------------------------------
# Modeles Pydantic
# ---------------------------------------------------------------------------

class FogAResult(BaseModel):
    score: float
    block: bool


class TransactionIn(BaseModel):
    tx_id: str
    card_id: str
    amount: float
    timestamp: int
    city: str
    atm_id: str = ""
    fog_a: FogAResult


class FogBResult(BaseModel):
    impossible_travel: bool
    distance_km: float
    speed_kmh: float
    previous_city: str | None
    time_delta_s: float | None


class DecisionReceived(BaseModel):
    tx_id: str
    decision: str
    reason: str
    details: str = ""


# ---------------------------------------------------------------------------
# Etat global
# ---------------------------------------------------------------------------
DB: sqlite3.Connection | None = None
backup_scorer: BehaviorScorer | None = None
_stop_sync = threading.Event()


# ---------------------------------------------------------------------------
# Logique geo
# ---------------------------------------------------------------------------

def check_travel(card_id: str, atm_id: str, city: str, ts: int) -> FogBResult:
    curr_coords = get_coordinates(atm_id, city)
    if curr_coords is None:
        return FogBResult(impossible_travel=False, distance_km=0.0,
                          speed_kmh=0.0, previous_city=None, time_delta_s=None)
    row = DB.execute(
        "SELECT last_atm_id, last_city, last_lat, last_lon, last_ts FROM card_locations WHERE card_id = ?",
        (card_id,),
    ).fetchone()
    if row is None:
        return FogBResult(impossible_travel=False, distance_km=0.0,
                          speed_kmh=0.0, previous_city=None, time_delta_s=None)
    prev_atm, prev_city, prev_lat, prev_lon, prev_ts = row
    dt = max(ts - prev_ts, 1)
    distance = haversine_km((prev_lat, prev_lon), curr_coords)
    speed = distance / (dt / 3600.0)
    impossible = (distance > MIN_DISTANCE_KM) and (speed > MAX_SPEED_KMH)
    prev_label = f"{prev_atm} ({prev_city})" if prev_atm else prev_city
    return FogBResult(impossible_travel=impossible, distance_km=round(distance, 2),
                      speed_kmh=round(speed, 2), previous_city=prev_label,
                      time_delta_s=float(dt))


def update_location(card_id: str, atm_id: str, city: str, ts: int):
    coords = get_coordinates(atm_id, city)
    if not coords:
        return
    lat, lon = coords
    DB.execute(
        "INSERT INTO card_locations (card_id, last_atm_id, last_city, last_lat, last_lon, last_ts) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(card_id) DO UPDATE SET "
        "last_atm_id=excluded.last_atm_id, last_city=excluded.last_city, "
        "last_lat=excluded.last_lat, last_lon=excluded.last_lon, last_ts=excluded.last_ts",
        (card_id, atm_id, city, lat, lon, ts),
    )
    DB.commit()


# ---------------------------------------------------------------------------
# File offline et sync
# ---------------------------------------------------------------------------

def _store_offline(tx_id, card_id, amount, city, timestamp,
                   fog_a_score, impossible_travel, decision_dict):
    DB.execute(
        "INSERT INTO offline_queue "
        "(tx_id, card_id, amount, city, timestamp, fog_a_score, "
        "impossible_travel, decision, reason, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tx_id, card_id, amount, city, timestamp, fog_a_score,
         int(impossible_travel), decision_dict["decision"], decision_dict["reason"],
         datetime.now(timezone.utc).isoformat()),
    )
    DB.commit()


def _sync_worker():
    log.info("Thread sync offline demarre (intervalle %ds)", SYNC_INTERVAL)
    while not _stop_sync.is_set():
        _stop_sync.wait(SYNC_INTERVAL)
        if _stop_sync.is_set():
            break
        try:
            rows = DB.execute(
                "SELECT id, tx_id, card_id, amount, city, timestamp, "
                "fog_a_score, impossible_travel, decision, reason, decided_at "
                "FROM offline_queue WHERE synced=0 ORDER BY id LIMIT 50"
            ).fetchall()
            if not rows:
                continue
            payload = [
                {"tx_id": r[1], "card_id": r[2], "amount": r[3], "city": r[4],
                 "timestamp": r[5], "fog_a_score": r[6], "impossible_travel": bool(r[7]),
                 "decision": r[8], "reason": r[9], "decided_at": r[10],
                 "source_node": NODE_ID}
                for r in rows
            ]
            ids = [r[0] for r in rows]
            with httpx.Client(verify=_ssl_ctx, timeout=5.0) as client:
                resp = client.post(
                    f"{FOG_C_URL}/sync_offline_decisions", json=payload,
                    headers=auth.auth_header("fog_c"),
                )
                resp.raise_for_status()
            DB.execute(
                "UPDATE offline_queue SET synced=1 WHERE id IN (%s)"
                % ",".join("?" * len(ids)), ids,
            )
            DB.commit()
            log.info("Sync offline OK : %d decisions envoyees a Fog C", len(ids))
        except Exception as exc:
            log.debug("Sync offline echec (%s) -- retry dans %ds", exc, SYNC_INTERVAL)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB, backup_scorer
    DB = sqlite3.connect("fog_b_locations.db", check_same_thread=False)
    DB.execute("PRAGMA journal_mode=WAL")
    DB.executescript(LOCATION_SCHEMA)
    DB.executescript(TRANSACTIONS_SCHEMA)
    DB.executescript(OFFLINE_QUEUE_SCHEMA)
    try:
        backup_scorer = BehaviorScorer("artifacts/model.onnx")
        log.info("Modele ONNX de secours charge (seuil=%.3f)", backup_scorer.threshold)
    except Exception as exc:
        log.warning("Modele ONNX de secours non disponible : %s", exc)
        backup_scorer = None
    log.info("Seuil vitesse : %.0f km/h | Distance min : %.0f km",
             MAX_SPEED_KMH, MIN_DISTANCE_KM)
    log.info("Architecture maillée active : A<->B, B<->C, securite mTLS+JWT")
    _stop_sync.clear()
    threading.Thread(target=_sync_worker, daemon=True).start()
    yield
    _stop_sync.set()
    DB.close()


app = FastAPI(title="Fog B -- Spatio-Temporel (Mesh + mTLS + JWT)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    pending = DB.execute(
        "SELECT COUNT(*) FROM offline_queue WHERE synced=0"
    ).fetchone()[0] if DB else 0
    return {"node": NODE_ID, "status": "ok",
            "backup_model_loaded": backup_scorer is not None,
            "offline_queue_pending": pending,
            "security": "mTLS+JWT"}


@app.post("/analyze")
async def analyze(
    tx: TransactionIn,
    _claims: dict = Depends(require_from("fog_a", auth)),
):
    """[Lien A<->B] Recoit la transaction de Fog A, effectue la geo,
    puis envoie le resultat geo DIRECTEMENT a Fog C [lien B<->C].
    Retourne aussi le resultat geo a Fog A.
    """
    t0 = time.perf_counter_ns()

    # 1. Analyse geo (par GAB ou ville)
    fog_b = check_travel(tx.card_id, tx.atm_id, tx.city, tx.timestamp)
    update_location(tx.card_id, tx.atm_id, tx.city, tx.timestamp)

    latency_ms = (time.perf_counter_ns() - t0) / 1e6
    travel_flag = "!! IMPOSSIBLE TRAVEL" if fog_b.impossible_travel else "OK"
    log.info("TX %-12s | %-20s -> %-8s | %.0f km | %.0f km/h | %s | %.2f ms",
             tx.tx_id, fog_b.previous_city or "--", tx.city,
             fog_b.distance_km, fog_b.speed_kmh, travel_flag, latency_ms)

    # Score comportemental : vient de Fog A, ou du modele de secours si A a envoye 0.0
    behavior_score = tx.fog_a.score
    if backup_scorer is not None and tx.fog_a.score == 0.0:
        history = fetch_history(DB, tx.card_id, tx.timestamp)
        bk_score, _, _ = backup_scorer.score(tx.amount, tx.timestamp, history)
        behavior_score = bk_score
        log.info("TX %-12s | Score IA de secours utilise : %.4f", tx.tx_id, behavior_score)

    # 2. [Lien B->C] : envoyer le rapport geo DIRECTEMENT a Fog C
    geo_payload = {
        "tx_id": tx.tx_id, "card_id": tx.card_id,
        "impossible_travel": fog_b.impossible_travel,
        "distance_km": fog_b.distance_km,
        "speed_kmh": fog_b.speed_kmh,
        "previous_city": fog_b.previous_city,
        "time_delta_s": fog_b.time_delta_s,
        "source_node": NODE_ID,
    }
    try:
        async with httpx.AsyncClient(verify=_ssl_ctx, timeout=PEER_TIMEOUT) as client:
            resp = await client.post(
                f"{FOG_C_URL}/submit_geo", json=geo_payload,
                headers=auth.auth_header("fog_c", tx.tx_id),
            )
            resp.raise_for_status()
            log.info("TX %-12s | [B->C] Rapport geo transmis directement a Fog C", tx.tx_id)
    except Exception as exc:
        log.warning("TX %-12s | [B->C] submit_geo echec (%s) -- C utilisera le /decide de A",
                    tx.tx_id, exc)

    # 3. Retourner le resultat geo a Fog A (Fog A s'occupe d'appeler /decide sur C)
    return {
        "tx_id": tx.tx_id,
        "fog_b": fog_b.model_dump(),
        "behavior_score_used": behavior_score,
        "latency_fog_b_ms": round(latency_ms, 2),
    }


@app.post("/receive_decision")
async def receive_decision(
    data: DecisionReceived,
    _claims: dict = Depends(require_from("fog_c", auth)),
):
    """[Lien C->B] Reçoit le broadcast de decision finale de Fog C.

    Fog C informe activement Fog B de la decision apres correlation,
    ce qui permet a Fog B de mettre a jour son contexte local si necessaire.
    """
    log.info("TX %-12s | [C->B] Broadcast reçu de Fog C : %s (%s)",
             data.tx_id, data.decision, data.reason)
    return {"status": "acknowledged", "node": NODE_ID}


@app.post("/sync_location")
async def sync_location(
    data: dict,
    _claims: dict = Depends(require_from("fog_a", auth)),
):
    """Mise a jour de position depuis Fog A (lien A<->B horizontal)."""
    card_id = data.get("card_id")
    atm_id = data.get("atm_id", "")
    city = data.get("city")
    ts = data.get("timestamp")
    if card_id and city and ts:
        update_location(card_id, atm_id, city, int(ts))
        return {"status": "synced"}
    raise HTTPException(status_code=400, detail="card_id, city et timestamp requis")


@app.get("/offline_queue")
def view_offline_queue(limit: int = 20):
    rows = DB.execute(
        "SELECT tx_id, card_id, amount, city, decision, reason, synced, decided_at "
        "FROM offline_queue ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"tx_id": r[0], "card_id": r[1], "amount": r[2], "city": r[3],
             "decision": r[4], "reason": r[5], "synced": bool(r[6]),
             "decided_at": r[7]} for r in rows]


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fog B -- Spatio-Temporel (Mesh + mTLS + JWT)")
    ap.add_argument("--port", type=int, default=CONFIG["port"])
    ap.add_argument("--host", default=CONFIG["host"])
    args = ap.parse_args()
    log.info("Demarrage Fog B sur %s:%d (HTTPS + mTLS + JWT)", args.host, args.port)
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_certfile=SEC["node_cert"],
        ssl_keyfile=SEC["node_key"],
        ssl_ca_certs=SEC["ca_cert"],
        log_level="warning",
    )
