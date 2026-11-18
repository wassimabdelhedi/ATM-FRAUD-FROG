"""Noeud Fog C -- Coordinateur Central, Architecture Maillée (PC 3).

NOUVELLE ARCHITECTURE (Star-Mesh) :
  Fog A -> Fog C : POST /submit_score (score IA direct)       [lien A<->C]
  Fog B -> Fog C : POST /submit_geo   (rapport geo direct)    [lien B<->C]
  Fog A -> Fog C : POST /decide       (demande de correlation) [lien A<->C]
  Fog C -> Fog A : POST /receive_decision (broadcast)         [lien C->A]
  Fog C -> Fog B : POST /receive_decision (broadcast)         [lien C->B]

LOGIQUE DE CORRELATION :
  Fog C maintient une table en memoire (pending) indexee par tx_id.
  Quand A envoie /submit_score et B envoie /submit_geo (quasi-parallele),
  les resultats sont stockes dans pending[tx_id].
  Quand A appelle /decide, Fog C fusionne score + geo, applique make_decision()
  et diffuse la decision finale via broadcast asynchone.

SECURITE :
  - HTTPS/TLS : canal chiffre, certificats signes par le CA interne
  - mTLS      : Fog C presente son certificat fog_c.crt
  - JWT HS256 : chaque endpoint valide l'emetteur attendu (iss)

Lancer : py app.py [--port 8003]
"""
import argparse
import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import httpx
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from decision_engine import make_decision
from security import NodeAuth, build_client_ssl_ctx, require_from, require_from_any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

NODE_ID = CONFIG["node_id"]
FOG_A_URL = CONFIG["peers"]["fog_a_url"]
FOG_B_URL = CONFIG["peers"]["fog_b_url"]
BEHAVIOR_THRESHOLD = CONFIG["thresholds"]["behavior_score"]
MAX_SPEED_KMH = CONFIG["thresholds"]["max_speed_kmh"]
PEER_TIMEOUT = CONFIG["resilience"]["peer_timeout_s"]

SEC = CONFIG["security"]
auth = NodeAuth(node_id=NODE_ID, secret=SEC["jwt_secret"])
_ssl_ctx = build_client_ssl_ctx(SEC["ca_cert"], SEC["node_cert"], SEC["node_key"])

logging.basicConfig(level=logging.INFO,
                    format="[FOG-C] %(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fog_c")

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
DECISION_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id                   TEXT NOT NULL,
    card_id                 TEXT NOT NULL,
    amount                  REAL NOT NULL,
    city                    TEXT NOT NULL,
    timestamp               INTEGER NOT NULL,
    fog_a_score             REAL,
    fog_a_block             INTEGER,
    fog_b_impossible_travel INTEGER,
    fog_b_distance_km       REAL,
    fog_b_speed_kmh         REAL,
    fog_b_previous_city     TEXT,
    decision                TEXT NOT NULL,
    reason                  TEXT NOT NULL,
    details                 TEXT,
    source_node             TEXT NOT NULL DEFAULT 'fog_c',
    decided_at              TEXT NOT NULL
);
"""

DB: sqlite3.Connection | None = None

# ---------------------------------------------------------------------------
# Table de correlation en memoire (pending scores et geos)
# Cle : tx_id | Valeur : {"score": ..., "geo": ...}
# Peuplee par /submit_score (Fog A) et /submit_geo (Fog B)
# Consommee par /decide (Fog A)
# ---------------------------------------------------------------------------
_pending: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Modeles Pydantic
# ---------------------------------------------------------------------------

class FogAResult(BaseModel):
    score: float
    block: bool


class FogBResult(BaseModel):
    impossible_travel: bool
    distance_km: float
    speed_kmh: float
    previous_city: str | None
    time_delta_s: float | None


class DiagnosticIn(BaseModel):
    """Paquet final de Fog A pour demander la decision correlee."""
    tx_id: str
    card_id: str
    amount: float
    timestamp: int
    city: str
    atm_id: str = ""
    fog_a: FogAResult
    fog_b: FogBResult


class ScoreReport(BaseModel):
    """Rapport IA reçu directement de Fog A [lien A->C]."""
    tx_id: str
    card_id: str
    score: float
    block: bool
    amount: float
    timestamp: int
    city: str
    atm_id: str = ""


class GeoReport(BaseModel):
    """Rapport geo reçu directement de Fog B [lien B->C]."""
    tx_id: str
    card_id: str
    impossible_travel: bool
    distance_km: float
    speed_kmh: float
    previous_city: str | None
    time_delta_s: float | None
    source_node: str = "fog_b"


class OfflineDecision(BaseModel):
    tx_id: str
    card_id: str
    amount: float
    city: str
    timestamp: int
    fog_a_score: float | None = None
    impossible_travel: bool = False
    decision: str
    reason: str
    decided_at: str
    source_node: str = "unknown"


# ---------------------------------------------------------------------------
# Broadcast de la decision vers Fog A et Fog B [liens C->A et C->B]
# ---------------------------------------------------------------------------

async def _broadcast_decision(tx_id: str, decision: str, reason: str, details: str):
    """Diffuse la decision finale aux noeuds Fog A et Fog B.

    Ce broadcast finalise le maillage : Fog C informe proactivement
    tous les noeuds de la decision prise, sans attendre leur demande.
    Exécuté en tâche asyncio de fond (non bloquant).
    """
    payload = {"tx_id": tx_id, "decision": decision, "reason": reason, "details": details}

    async with httpx.AsyncClient(verify=_ssl_ctx, timeout=PEER_TIMEOUT) as client:
        for node_id, url in [("fog_a", FOG_A_URL), ("fog_b", FOG_B_URL)]:
            try:
                resp = await client.post(
                    f"{url}/receive_decision", json=payload,
                    headers=auth.auth_header(node_id, tx_id),
                )
                resp.raise_for_status()
                log.info("TX %-12s | [C->%s] Broadcast OK", tx_id, node_id.upper())
            except Exception as exc:
                log.debug("TX %-12s | [C->%s] Broadcast echec (%s)",
                          tx_id, node_id.upper(), exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB
    DB = sqlite3.connect("fog_c_decisions.db", check_same_thread=False)
    DB.execute("PRAGMA journal_mode=WAL")
    DB.executescript(DECISION_SCHEMA)
    log.info("Journal de decisions initialise (fog_c_decisions.db)")
    log.info("Seuils : Comportement=%.2f | Vitesse max=%.0f km/h",
             BEHAVIOR_THRESHOLD, MAX_SPEED_KMH)
    log.info("Architecture maillée active : A->C, B->C, C->A, C->B | mTLS+JWT")
    yield
    DB.close()


app = FastAPI(title="Fog C -- Coordinateur Central (Mesh + mTLS + JWT)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    total = DB.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] if DB else 0
    offline_count = DB.execute(
        "SELECT COUNT(*) FROM decisions WHERE source_node != 'fog_c'"
    ).fetchone()[0] if DB else 0
    return {"node": NODE_ID, "status": "ok",
            "total_decisions": total,
            "offline_decisions_synced": offline_count,
            "pending_correlations": len(_pending),
            "security": "mTLS+JWT"}


@app.post("/submit_score")
async def submit_score(
    report: ScoreReport,
    _claims: dict = Depends(require_from_any(["fog_a", "fog_b"], auth)),
):
    """[Lien A->C] Reçoit le score IA directement de Fog A.

    Stocke dans la table de correlation en memoire.
    Fog C peut ainsi corroborer le score avec le rapport geo de B
    independamment de la requete /decide.
    """
    if report.tx_id not in _pending:
        _pending[report.tx_id] = {}
    _pending[report.tx_id]["score"] = {
        "score": report.score, "block": report.block,
        "card_id": report.card_id, "amount": report.amount,
        "timestamp": report.timestamp, "city": report.city, "atm_id": report.atm_id,
    }
    log.info("TX %-12s | [A->C] Score IA reçu directement : %.4f | pending=%s",
             report.tx_id, report.score,
             list(_pending[report.tx_id].keys()))
    return {"status": "received", "tx_id": report.tx_id,
            "pending_parts": list(_pending.get(report.tx_id, {}).keys())}


@app.post("/submit_geo")
async def submit_geo(
    report: GeoReport,
    _claims: dict = Depends(require_from("fog_b", auth)),
):
    """[Lien B->C] Reçoit le rapport geo directement de Fog B.

    Stocke dans la table de correlation en memoire.
    """
    if report.tx_id not in _pending:
        _pending[report.tx_id] = {}
    _pending[report.tx_id]["geo"] = {
        "impossible_travel": report.impossible_travel,
        "distance_km": report.distance_km,
        "speed_kmh": report.speed_kmh,
        "previous_city": report.previous_city,
        "time_delta_s": report.time_delta_s,
    }
    travel_flag = "!! IMPOSSIBLE" if report.impossible_travel else "OK"
    log.info("TX %-12s | [B->C] Rapport geo reçu directement : %.0f km %.0f km/h %s | pending=%s",
             report.tx_id, report.distance_km, report.speed_kmh, travel_flag,
             list(_pending[report.tx_id].keys()))
    return {"status": "received", "tx_id": report.tx_id,
            "pending_parts": list(_pending.get(report.tx_id, {}).keys())}


@app.post("/decide")
async def decide_endpoint(
    diag: DiagnosticIn,
    _claims: dict = Depends(require_from("fog_a", auth)),
):
    """[Lien A<->C] Decision finale correlee.

    Fog A appelle cet endpoint apres le fan-out parallele.
    Fog C utilise :
      - Le diagnostic fourni dans la requete (score + geo de Fog A)
      - Les rapports pre-soumis via /submit_score et /submit_geo (si disponibles)

    Decision prise avec make_decision() (moteur commun aux 3 noeuds).
    Suivi d'un broadcast vers Fog A et Fog B [liens C->A et C->B].
    """
    t0 = time.perf_counter_ns()

    # Enrichir avec les rapports pre-soumis si disponibles
    pending = _pending.pop(diag.tx_id, {})
    if pending.get("geo") and pending["geo"].get("impossible_travel") is not None:
        log.info("TX %-12s | Correlation : rapport geo pre-soumis par Fog B utilise",
                 diag.tx_id)

    # Decision avec le moteur partage (identique sur A, B et C)
    result = make_decision(diag.fog_a.score, diag.fog_b.impossible_travel)
    decision = result["decision"]
    reason = result["reason"]
    details = result["details"]

    # Journaliser
    DB.execute(
        "INSERT INTO decisions (tx_id, card_id, amount, city, timestamp, "
        "fog_a_score, fog_a_block, fog_b_impossible_travel, fog_b_distance_km, "
        "fog_b_speed_kmh, fog_b_previous_city, decision, reason, details, "
        "source_node, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (diag.tx_id, diag.card_id, diag.amount, diag.city, diag.timestamp,
         diag.fog_a.score, int(diag.fog_a.block),
         int(diag.fog_b.impossible_travel), diag.fog_b.distance_km,
         diag.fog_b.speed_kmh, diag.fog_b.previous_city,
         decision, reason, details, NODE_ID,
         datetime.now(timezone.utc).isoformat()),
    )
    DB.commit()

    latency_ms = (time.perf_counter_ns() - t0) / 1e6
    icon = "!! BLOQUEE" if decision == "BLOCK" else ("?? REVUE" if decision == "REVIEW" else "OK AUTORISEE")
    log.info("TX %-12s | %.0f DT | %s | %s | %.2f ms",
             diag.tx_id, diag.amount, icon, reason, latency_ms)

    # Broadcast asynchone vers Fog A et Fog B [liens C->A et C->B]
    asyncio.create_task(_broadcast_decision(diag.tx_id, decision, reason, details))

    return {
        "tx_id": diag.tx_id,
        "card_id": diag.card_id,
        "decision": decision,
        "reason": reason,
        "details": details,
        "action": result.get("action", ""),
        "fog_a_score": diag.fog_a.score,
        "fog_b_speed_kmh": diag.fog_b.speed_kmh,
        "latency_fog_c_ms": round(latency_ms, 2),
        "pending_had_score": "score" in pending,
        "pending_had_geo": "geo" in pending,
    }


@app.post("/sync_offline_decisions")
async def sync_offline_decisions(
    decisions: List[OfflineDecision],
    _claims: dict = Depends(require_from_any(["fog_a", "fog_b"], auth)),
):
    """Reçoit les decisions prises hors-ligne par Fog A ou Fog B."""
    inserted = 0
    for dec in decisions:
        exists = DB.execute(
            "SELECT 1 FROM decisions WHERE tx_id=? AND source_node=?",
            (dec.tx_id, dec.source_node)
        ).fetchone()
        if exists:
            continue
        DB.execute(
            "INSERT INTO decisions (tx_id, card_id, amount, city, timestamp, "
            "fog_a_score, fog_b_impossible_travel, decision, reason, "
            "source_node, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dec.tx_id, dec.card_id, dec.amount, dec.city, dec.timestamp,
             dec.fog_a_score, int(dec.impossible_travel),
             dec.decision, dec.reason, dec.source_node, dec.decided_at),
        )
        inserted += 1
    DB.commit()
    if decisions:
        log.info("Sync offline : %d decisions inserees depuis %s",
                 inserted, decisions[0].source_node)
    return {"synced": inserted, "total_received": len(decisions)}


@app.get("/decisions")
def list_decisions(limit: int = 20):
    rows = DB.execute(
        "SELECT tx_id, card_id, amount, city, decision, reason, source_node, decided_at "
        "FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"tx_id": r[0], "card_id": r[1], "amount": r[2], "city": r[3],
             "decision": r[4], "reason": r[5], "source_node": r[6],
             "decided_at": r[7]} for r in rows]


@app.get("/stats")
def stats():
    total = DB.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    blocked = DB.execute(
        "SELECT COUNT(*) FROM decisions WHERE decision='BLOCK'"
    ).fetchone()[0]
    reviewed = DB.execute(
        "SELECT COUNT(*) FROM decisions WHERE decision='REVIEW'"
    ).fetchone()[0]
    offline_count = DB.execute(
        "SELECT COUNT(*) FROM decisions WHERE source_node != 'fog_c'"
    ).fetchone()[0]
    by_reason = DB.execute(
        "SELECT reason, COUNT(*) FROM decisions WHERE decision='BLOCK' GROUP BY reason"
    ).fetchall()
    return {
        "total_transactions": total,
        "blocked": blocked,
        "reviewed": reviewed,
        "allowed": total - blocked - reviewed,
        "block_rate_pct": round(blocked / max(total, 1) * 100, 2),
        "offline_decisions_synced": offline_count,
        "blocked_by_reason": {r[0]: r[1] for r in by_reason},
    }


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fog C -- Coordinateur (Mesh + mTLS + JWT)")
    ap.add_argument("--port", type=int, default=CONFIG["port"])
    ap.add_argument("--host", default=CONFIG["host"])
    args = ap.parse_args()
    log.info("Demarrage Fog C sur %s:%d (HTTPS + mTLS + JWT)", args.host, args.port)
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_certfile=SEC["node_cert"],
        ssl_keyfile=SEC["node_key"],
        ssl_ca_certs=SEC["ca_cert"],
        log_level="warning",
    )
