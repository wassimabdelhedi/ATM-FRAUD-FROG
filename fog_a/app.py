"""Noeud Fog A -- Analyse Comportementale IA, Architecture Maillée (PC 1).

NOUVELLE ARCHITECTURE (Star-Mesh) :
  GAB --> Fog A : reçoit la transaction brute
  Fog A --> Fog B (en parallele) : demande l'analyse geographique  [lien A<->B]
  Fog A --> Fog C (en parallele) : signale directement le score IA [lien A<->C]
  Fog B --> Fog C               : transmet son rapport geo direct  [lien B<->C]
  Fog A --> Fog C               : demande la decision correlee finale
  Fog C --> Fog A et Fog B      : diffuse (broadcast) la decision finale

SECURITE :
  - HTTPS/TLS : canal chiffre avec certificats auto-signes (Root CA interne)
  - mTLS      : Fog A presente son certificat fog_a.crt a chaque appel
  - JWT HS256 : chaque requete inter-noeuds porte un token signe (60s d'expiration)
  - Endpoints proteges : /receive_decision accepte uniquement les tokens de fog_c

Lancer : py app.py [--port 8001]
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
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from decision_engine import make_decision
from geo import CITIES, ATMS, get_coordinates, haversine_km
from inference import BehaviorScorer, ensure_schema, fetch_history
from security import NodeAuth, build_client_ssl_ctx, require_from

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

NODE_ID = CONFIG["node_id"]
FOG_B_URL = CONFIG["peers"]["fog_b_url"]
FOG_C_URL = CONFIG["peers"]["fog_c_url"]
SYNC_INTERVAL = CONFIG["resilience"]["sync_interval_s"]
PEER_TIMEOUT = CONFIG["resilience"]["peer_timeout_s"]
MIN_DISTANCE_KM = CONFIG["thresholds"]["min_distance_km"]
MAX_SPEED_KMH = CONFIG["thresholds"]["max_speed_kmh"]

SEC = CONFIG["security"]
auth = NodeAuth(node_id=NODE_ID, secret=SEC["jwt_secret"])
_ssl_ctx = build_client_ssl_ctx(SEC["ca_cert"], SEC["node_cert"], SEC["node_key"])

logging.basicConfig(level=logging.INFO,
                    format="[FOG-A] %(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fog_a")

# ---------------------------------------------------------------------------
# Modeles Pydantic
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    tx_id: str
    card_id: str
    amount: float
    timestamp: int
    city: str
    atm_id: str = ""


class FogAResult(BaseModel):
    score: float
    block: bool


class TransactionToFogB(BaseModel):
    tx_id: str
    card_id: str
    amount: float
    timestamp: int
    city: str
    atm_id: str
    fog_a: FogAResult


class DecisionReceived(BaseModel):
    """Payload reçu via POST /receive_decision depuis Fog C."""
    tx_id: str
    decision: str
    reason: str
    details: str = ""


# ---------------------------------------------------------------------------
# Schemas SQLite (transactions + localisation backup + file offline)
# ---------------------------------------------------------------------------
OFFLINE_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS offline_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id       TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    amount      REAL NOT NULL,
    city        TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    fog_a_score REAL,
    impossible_travel INTEGER,
    decision    TEXT NOT NULL,
    reason      TEXT NOT NULL,
    decided_at  TEXT NOT NULL,
    synced      INTEGER NOT NULL DEFAULT 0
);
"""

LOCATION_SCHEMA_A = """
CREATE TABLE IF NOT EXISTS card_locations (
    card_id     TEXT PRIMARY KEY,
    last_atm_id TEXT,
    last_city   TEXT NOT NULL,
    last_lat    REAL NOT NULL,
    last_lon    REAL NOT NULL,
    last_ts     INTEGER NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Etat global
# ---------------------------------------------------------------------------
scorer: BehaviorScorer | None = None
db: sqlite3.Connection | None = None
_stop_sync = threading.Event()


# ---------------------------------------------------------------------------
# Geo de secours local (si Fog B injoignable)
# ---------------------------------------------------------------------------

def _local_geo_check(card_id: str, atm_id: str, city: str, ts: int) -> dict:
    """Replication locale de la logique Fog B pour le mode secours."""
    curr_coords = get_coordinates(atm_id, city)
    if curr_coords is None:
        return {"impossible_travel": False, "distance_km": 0.0,
                "speed_kmh": 0.0, "previous_city": None, "time_delta_s": None}
    row = db.execute(
        "SELECT last_atm_id, last_city, last_lat, last_lon, last_ts FROM card_locations WHERE card_id = ?",
        (card_id,),
    ).fetchone()
    if row is None:
        return {"impossible_travel": False, "distance_km": 0.0,
                "speed_kmh": 0.0, "previous_city": None, "time_delta_s": None}
    prev_atm, prev_city, prev_lat, prev_lon, prev_ts = row
    dt = max(ts - prev_ts, 1)
    distance = haversine_km((prev_lat, prev_lon), curr_coords)
    speed = distance / (dt / 3600.0)
    impossible = (distance > MIN_DISTANCE_KM) and (speed > MAX_SPEED_KMH)
    prev_label = f"{prev_atm} ({prev_city})" if prev_atm else prev_city
    return {"impossible_travel": impossible, "distance_km": round(distance, 2),
            "speed_kmh": round(speed, 2), "previous_city": prev_label,
            "time_delta_s": float(dt)}


def _update_local_location(card_id: str, atm_id: str, city: str, ts: int):
    coords = get_coordinates(atm_id, city)
    if not coords:
        return
    lat, lon = coords
    db.execute(
        "INSERT INTO card_locations (card_id, last_atm_id, last_city, last_lat, last_lon, last_ts) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(card_id) DO UPDATE SET "
        "last_atm_id=excluded.last_atm_id, last_city=excluded.last_city, "
        "last_lat=excluded.last_lat, last_lon=excluded.last_lon, last_ts=excluded.last_ts",
        (card_id, atm_id, city, lat, lon, ts),
    )
    db.commit()


# ---------------------------------------------------------------------------
# File offline et sync vers Fog C
# ---------------------------------------------------------------------------

def _store_offline(tx_id, card_id, amount, city, timestamp,
                   fog_a_score, impossible_travel, decision_dict):
    db.execute(
        "INSERT INTO offline_queue "
        "(tx_id, card_id, amount, city, timestamp, fog_a_score, "
        "impossible_travel, decision, reason, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tx_id, card_id, amount, city, timestamp, fog_a_score,
         int(impossible_travel), decision_dict["decision"], decision_dict["reason"],
         datetime.now(timezone.utc).isoformat()),
    )
    db.commit()


def _sync_worker():
    log.info("Thread sync offline demarre (intervalle %ds)", SYNC_INTERVAL)
    while not _stop_sync.is_set():
        _stop_sync.wait(SYNC_INTERVAL)
        if _stop_sync.is_set():
            break
        try:
            rows = db.execute(
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
            db.execute(
                "UPDATE offline_queue SET synced=1 WHERE id IN (%s)"
                % ",".join("?" * len(ids)), ids,
            )
            db.commit()
            log.info("Sync offline OK : %d decisions envoyees a Fog C", len(ids))
        except Exception as exc:
            log.debug("Sync offline echec (%s) -- retry dans %ds", exc, SYNC_INTERVAL)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, db
    scorer = BehaviorScorer("artifacts/model.onnx")
    db = sqlite3.connect("fog_a_transactions.db", check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    ensure_schema(db)
    db.executescript(OFFLINE_QUEUE_SCHEMA)
    db.executescript(LOCATION_SCHEMA_A)
    log.info("Modele ONNX charge (seuil=%.3f | features %s)",
             scorer.threshold, scorer.metadata.get("features_version"))
    log.info("Architecture maillée active : A<->B, A<->C, securite mTLS+JWT")
    _stop_sync.clear()
    threading.Thread(target=_sync_worker, daemon=True).start()
    yield
    _stop_sync.set()
    db.close()


app = FastAPI(title="Fog A -- IA Comportementale (Mesh + mTLS + JWT)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    pending = db.execute(
        "SELECT COUNT(*) FROM offline_queue WHERE synced=0"
    ).fetchone()[0] if db else 0
    return {"node": NODE_ID, "status": "ok",
            "threshold": scorer.threshold if scorer else None,
            "offline_queue_pending": pending,
            "security": "mTLS+JWT"}


@app.post("/transaction")
async def process_transaction(tx: TransactionRequest):
    """Point d'entree principal (GAB).

    NOUVELLE ARCHITECTURE MAILLÉE :
    1. Inference IA (Random Forest ONNX)
    2. Fan-out PARALLELE :
       A -> B : POST /analyze (geo check)       [lien horizontal A<->B]
       A -> C : POST /submit_score (score IA)   [lien direct A<->C]
    3. Fog B -> C : POST /submit_geo (direct)   [lien direct B<->C]
    4. A -> C : POST /decide (decision finale)  [lien A<->C]
    5. C -> A et B : broadcast decision         [liens C->A et C->B]
    """
    t0 = time.perf_counter_ns()

    # 1. Inference IA
    history = fetch_history(db, tx.card_id, tx.timestamp)
    score, block, _features = scorer.score(tx.amount, tx.timestamp, history)
    db.execute("INSERT INTO transactions(card_id, ts, amount) VALUES (?, ?, ?)",
               (tx.card_id, tx.timestamp, tx.amount))
    db.commit()

    latency_a_ms = (time.perf_counter_ns() - t0) / 1e6
    log.info("TX %-12s | carte %-10s | %.0f DT | score=%.4f | %.2f ms",
             tx.tx_id, tx.card_id, tx.amount, score, latency_a_ms)

    fog_a_result = FogAResult(score=round(score, 4), block=block)
    payload_b = TransactionToFogB(tx_id=tx.tx_id, card_id=tx.card_id,
                                   amount=tx.amount, timestamp=tx.timestamp,
                                   city=tx.city, atm_id=tx.atm_id,
                                   fog_a=fog_a_result)
    score_payload = {
        "tx_id": tx.tx_id, "card_id": tx.card_id,
        "score": score, "block": block,
        "amount": tx.amount, "timestamp": tx.timestamp,
        "city": tx.city, "atm_id": tx.atm_id,
    }

    # 2. Fan-out parallele : A->B (geo) + A->C (submit score)
    fog_b_data = None
    fog_b_reachable = False

    async with httpx.AsyncClient(verify=_ssl_ctx, timeout=PEER_TIMEOUT) as client:
        results = await asyncio.gather(
            client.post(f"{FOG_B_URL}/analyze",
                        json=payload_b.model_dump(),
                        headers=auth.auth_header("fog_b", tx.tx_id)),
            client.post(f"{FOG_C_URL}/submit_score",
                        json=score_payload,
                        headers=auth.auth_header("fog_c", tx.tx_id)),
            return_exceptions=True,
        )

    res_b, res_c_score = results

    if isinstance(res_b, Exception):
        log.warning("TX %-12s | Fog B injoignable (%s) -- geo locale activee",
                    tx.tx_id, res_b)
        geo = _local_geo_check(tx.card_id, tx.atm_id, tx.city, tx.timestamp)
        _update_local_location(tx.card_id, tx.atm_id, tx.city, tx.timestamp)
        fog_b_data = {"fog_b": geo}
    else:
        try:
            res_b.raise_for_status()
            fog_b_data = res_b.json()
            fog_b_reachable = True
            log.info("TX %-12s | [A<->B] Fog B OK -- travel=%s",
                     tx.tx_id, fog_b_data.get("fog_b", {}).get("impossible_travel"))
        except Exception as exc:
            log.warning("TX %-12s | Fog B erreur (%s) -- geo locale activee",
                        tx.tx_id, exc)
            geo = _local_geo_check(tx.card_id, tx.atm_id, tx.city, tx.timestamp)
            fog_b_data = {"fog_b": geo}

    if isinstance(res_c_score, Exception):
        log.debug("TX %-12s | [A->C] submit_score echec (%s)", tx.tx_id, res_c_score)
    else:
        log.info("TX %-12s | [A->C] Score IA transmis directement a Fog C", tx.tx_id)

    # 3. Extraire geo result
    fog_b_result = fog_b_data.get("fog_b", {}) if fog_b_data else {}
    impossible_travel = fog_b_result.get("impossible_travel", False)

    # 4. Demander la decision finale a Fog C (qui a recu score via A->C et geo via B->C)
    decide_payload = {
        "tx_id": tx.tx_id, "card_id": tx.card_id,
        "amount": tx.amount, "timestamp": tx.timestamp,
        "city": tx.city, "atm_id": tx.atm_id,
        "fog_a": fog_a_result.model_dump(),
        "fog_b": fog_b_result if fog_b_result else {
            "impossible_travel": False, "distance_km": 0.0,
            "speed_kmh": 0.0, "previous_city": None, "time_delta_s": None,
        },
    }

    fog_c_reachable = False
    source = "local_fallback"
    final_decision = None

    try:
        async with httpx.AsyncClient(verify=_ssl_ctx, timeout=PEER_TIMEOUT) as client:
            resp = await client.post(
                f"{FOG_C_URL}/decide",
                json=decide_payload,
                headers=auth.auth_header("fog_c", tx.tx_id),
            )
            resp.raise_for_status()
            final_decision = resp.json()
            fog_c_reachable = True
            source = "fog_c"
            log.info("TX %-12s | [A<->C] Decision Fog C : %s",
                     tx.tx_id, final_decision.get("decision"))
    except Exception as exc:
        log.warning("TX %-12s | Fog C injoignable (%s) -- decision locale Fog A",
                    tx.tx_id, exc)
        final_decision = make_decision(score, impossible_travel)
        _store_offline(tx.tx_id, tx.card_id, tx.amount, tx.city,
                       tx.timestamp, score, impossible_travel, final_decision)
        log.info("TX %-12s | Decision locale A : %s (%s)",
                 tx.tx_id, final_decision["decision"], final_decision["reason"])

    return {
        "tx_id": tx.tx_id,
        "decision": final_decision.get("decision", "ALLOW"),
        "reason": final_decision.get("reason", ""),
        "details": final_decision.get("details", ""),
        "action": final_decision.get("action", ""),
        "fog_a": fog_a_result.model_dump(),
        "fog_b_reachable": fog_b_reachable,
        "fog_c_reachable": fog_c_reachable,
        "decision_source": source,
        "latency_fog_a_ms": round(latency_a_ms, 2),
        "security": "mTLS+JWT",
    }


@app.post("/receive_decision")
async def receive_decision(
    data: DecisionReceived,
    _claims: dict = Depends(require_from("fog_c", auth)),
):
    """[Lien C->A] Reçoit le broadcast de decision finale de Fog C.

    Ce lien finalise l'architecture maillée : Fog C informe activement Fog A
    de la decision prise, sans attendre que Fog A la demande.
    """
    log.info("TX %-12s | [C->A] Broadcast reçu de Fog C : %s (%s)",
             data.tx_id, data.decision, data.reason)
    return {"status": "acknowledged", "node": NODE_ID}


@app.get("/offline_queue")
def view_offline_queue(limit: int = 20):
    rows = db.execute(
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
    ap = argparse.ArgumentParser(description="Fog A -- Analyse IA (Mesh + mTLS + JWT)")
    ap.add_argument("--port", type=int, default=CONFIG["port"])
    ap.add_argument("--host", default=CONFIG["host"])
    args = ap.parse_args()
    log.info("Demarrage Fog A sur %s:%d (HTTPS + mTLS + JWT)", args.host, args.port)
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_certfile=SEC["node_cert"],
        ssl_keyfile=SEC["node_key"],
        ssl_ca_certs=SEC["ca_cert"],    # mTLS : verifier les clients
        log_level="warning",
    )
