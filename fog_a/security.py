"""Module de securite inter-noeuds : JWT (HS256) + helpers SSL/mTLS.

Fonctionnement :
  - Chaque appel HTTP entre noeuds porte un jeton JWT signe (Authorization: Bearer ...).
  - Le JWT contient : emetteur (iss), destinataire (aud), id de transaction, expiration.
  - Expiration de 60 secondes : empeche les attaques par rejeu (replay attack).
  - mTLS : chaque client httpx presente son propre certificat TLS lors de la connexion.

Usage dans app.py :
    from security import NodeAuth, require_from, build_client_ssl_ctx

    auth = NodeAuth(node_id="fog_a", secret=CONFIG["jwt_secret"])

    # Generer un token pour appeler fog_b :
    headers = auth.auth_header(audience="fog_b", tx_id=tx.tx_id)

    # Proteger un endpoint (ne repond qu'a fog_a) :
    @app.post("/analyze")
    async def analyze(tx: ..., _claims=Depends(require_from("fog_a", auth))):
        ...
"""
import logging
import ssl
import time
from typing import Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("security")

ALGORITHM = "HS256"
TOKEN_EXPIRY_S = 60   # 60 secondes -- anti-replay

# ---------------------------------------------------------------------------
# NodeAuth : generation et validation des jetons JWT
# ---------------------------------------------------------------------------

class NodeAuth:
    """Gestionnaire JWT pour un noeud Fog."""

    def __init__(self, node_id: str, secret: str):
        self.node_id = node_id
        self.secret = secret

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def make_token(self, audience: str, tx_id: str = "") -> str:
        """Genere un JWT signe (HS256) avec expiration de 60s."""
        now = int(time.time())
        payload = {
            "iss": self.node_id,         # Emetteur (ex: "fog_a")
            "aud": audience,              # Destinataire (ex: "fog_b")
            "tx_id": tx_id,              # Identifiant de la transaction (tracabilite)
            "iat": now,                   # Instant d'emission
            "exp": now + TOKEN_EXPIRY_S,  # Expiration (anti-replay)
        }
        token = jwt.encode(payload, self.secret, algorithm=ALGORITHM)
        log.info("[SEC-JWT-OUT] Token genere: %s -> %s | tx_id=%s | exp=+%ds",
                 self.node_id, audience, tx_id or "none", TOKEN_EXPIRY_S)
        return token

    def auth_header(self, audience: str, tx_id: str = "") -> dict:
        """Retourne l'en-tete HTTP Authorization pour un appel inter-noeuds."""
        return {"Authorization": f"Bearer {self.make_token(audience, tx_id)}"}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def verify_token(self, token: str, expected_issuer: str) -> dict:
        """Valide un JWT entrant.

        Verifie :
          1. La signature (cle secrete partagee).
          2. L'audience (aud == node_id de ce noeud).
          3. L'expiration (exp > maintenant).
          4. L'emetteur (iss == expected_issuer).

        Raises jwt.InvalidTokenError si l'un des controles echoue.
        """
        decoded = jwt.decode(
            token,
            self.secret,
            algorithms=[ALGORITHM],
            audience=self.node_id,   # Rejet si le token n'est pas destine a ce noeud
        )
        actual_issuer = decoded.get("iss", "")
        if actual_issuer != expected_issuer:
            log.warning("[SEC-ALERT] Emetteur suspect! Attendu='%s', Reçu='%s'",
                        expected_issuer, actual_issuer)
            raise jwt.InvalidTokenError(
                f"Emetteur invalide : attendu '{expected_issuer}', recu '{actual_issuer}'"
            )
        log.info("[SEC-JWT-IN]  Token VALIDE : emetteur='%s' -> dest='%s' | tx_id=%s",
                 actual_issuer, self.node_id, decoded.get("tx_id", "none"))
        return decoded


# ---------------------------------------------------------------------------
# Dependance FastAPI : valide le JWT sur les endpoints proteges
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=True)


def require_from_any(expected_issuers: list, auth: "NodeAuth") -> Callable:
    """Retourne une dependance FastAPI acceptant plusieurs emetteurs valides.

    Usage (endpoint appele par fog_a OU fog_b) :
        @app.post("/sync_offline_decisions")
        async def sync(_, _claims=Depends(require_from_any(["fog_a","fog_b"], auth))):
            ...
    """
    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    ) -> dict:
        last_exc: Exception = Exception("Token invalide")
        for issuer in expected_issuers:
            try:
                return auth.verify_token(credentials.credentials, issuer)
            except jwt.ExpiredSignatureError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token JWT expire (delai de 60s depasse -- attaque par rejeu rejetee)",
                )
            except jwt.InvalidAudienceError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token JWT non destine a ce noeud (audience invalide)",
                )
            except jwt.InvalidTokenError as exc:
                last_exc = exc
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token JWT invalide : {last_exc}",
        )
    return _dependency


def require_from(expected_issuer: str, auth: "NodeAuth") -> Callable:
    """Retourne une dependance FastAPI qui valide le JWT et l'emetteur.

    Usage :
        @app.post("/analyze")
        async def analyze(tx: ..., _=Depends(require_from("fog_a", auth))):
            ...
    """
    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    ) -> dict:
        try:
            return auth.verify_token(credentials.credentials, expected_issuer)
        except jwt.ExpiredSignatureError:
            log.warning("[SEC-REJET] Token JWT expire (> %ds)", TOKEN_EXPIRY_S)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token JWT expire (delai de 60s depasse -- attaque par rejeu rejetee)",
            )
        except jwt.InvalidAudienceError:
            log.warning("[SEC-REJET] Token avec mauvaise audience (non destine a ce noeud)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token JWT non destine a ce noeud (audience invalide)",
            )
        except jwt.InvalidTokenError as exc:
            log.warning("[SEC-REJET] Token JWT invalide/rejete : %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token JWT invalide : {exc}",
            )
    return _dependency


# ---------------------------------------------------------------------------
# SSL / mTLS : contexte pour httpx (client) et uvicorn (serveur)
# ---------------------------------------------------------------------------

def build_client_ssl_ctx(ca_cert: str, client_cert: str, client_key: str) -> ssl.SSLContext:
    """Cree un contexte SSL client pour httpx avec mTLS.

    - ca_cert    : chemin vers ca.crt (pour verifier le serveur)
    - client_cert: chemin vers fog_X.crt (certificat du noeud appelant)
    - client_key : chemin vers fog_X.key (cle privee du noeud appelant)

    Ce contexte garantit :
      - Le chiffrement TLS du canal (confidentialite des donnees).
      - La verification du certificat serveur (evite le MITM).
      - La presentation du certificat client (mTLS -- authentication mutuelle).
    """
    ctx = ssl.create_default_context(cafile=ca_cert)
    ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False   # Desactive pour les IPs sur le LAN ; activer en prod
    return ctx
