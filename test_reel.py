
import httpx
import json
import time

BASE_A = "https://127.0.0.1:8001"
BASE_B = "https://127.0.0.1:8002"
BASE_C = "https://127.0.0.1:8003"
V = False   # verify=False (certificat auto-signe, usage local uniquement)


def sep(titre):
    print()
    print("=" * 60)
    print("  " + titre)
    print("=" * 60)


def affiche(label, d):
    if d is None:
        print(f"  {label} -> PAS DE REPONSE")
        return
    dec    = d.get("decision", "?")
    reason = d.get("reason",   "?")
    src    = d.get("decision_source", "?")
    fog_b  = d.get("fog_b_reachable", "?")
    fog_c  = d.get("fog_c_reachable",  "?")
    score  = d.get("fog_a", {}).get("score", "?")
    icon   = "BLOQUE  !!" if dec == "BLOCK" else ("REVUE   ??" if dec == "REVIEW" else "AUTORISE OK")
    print(f"  [{icon}]  {dec}")
    print(f"  Raison          : {reason}")
    print(f"  Source decision : {src}")
    print(f"  Score IA        : {score}")
    print(f"  Fog B joignable : {fog_b}")
    print(f"  Fog C joignable : {fog_c}")


def post(tx):
    try:
        r = httpx.post(f"{BASE_A}/transaction", json=tx, timeout=12, verify=V)
        return r.json()
    except Exception as e:
        print(f"  ERREUR : {e}")
        return None


def get(url):
    try:
        r = httpx.get(url, timeout=5, verify=V)
        return r.json()
    except Exception as e:
        print(f"  ERREUR : {e}")
        return None


# ============================================================
# ETAPE 1 : Health check
# ============================================================
sep("ETAPE 1 : Verification des 3 noeuds")
for port, nom in [(8001, "Fog A"), (8002, "Fog B"), (8003, "Fog C")]:
    try:
        r = httpx.get(f"https://127.0.0.1:{port}/health", timeout=4, verify=V)
        d = r.json()
        sec = d.get("security", "?")
        q   = d.get("offline_queue_pending", 0)
        print(f"  {nom} (:{port}) -> OK | securite={sec} | file_offline={q}")
    except Exception as e:
        print(f"  {nom} (:{port}) -> HORS LIGNE : {e}")

# ============================================================
# ETAPE 2 : Transaction normale
# ============================================================
sep("ETAPE 2 : Transaction normale (ALLOW attendu)")
print("  Carte : CARTE_AHMED | Ville : Tunis | Montant : 150 DT")
r1 = post({
    "tx_id": "TEST_01", "card_id": "CARTE_AHMED",
    "amount": 150.0, "timestamp": 1800000000,
    "city": "Tunis", "atm_id": "ATM_TUNIS_01"
})
affiche("TEST_01", r1)

time.sleep(1)

# ============================================================
# ETAPE 3 : Voyage impossible inter-villes
# ============================================================
sep("ETAPE 3 : Voyage impossible Tunis->Sfax en 5min (BLOCK attendu)")
print("  Meme carte CARTE_AHMED | 5min apres | 270 km = 3240 km/h")
r2 = post({
    "tx_id": "TEST_02", "card_id": "CARTE_AHMED",
    "amount": 200.0, "timestamp": 1800000300,
    "city": "Sfax", "atm_id": "ATM_SFAX_01"
})
affiche("TEST_02", r2)

time.sleep(1)

# ============================================================
# ETAPE 4 : Voyage impossible intra-ville
# ============================================================
sep("ETAPE 4 : Voyage intra-ville Tunis ATM01->ATM03 en 2min (BLOCK attendu)")
print("  Carte CARTE_INTRA | 1ere TX : ATM_TUNIS_01 (centre)")
r3a = post({
    "tx_id": "TEST_03A", "card_id": "CARTE_INTRA",
    "amount": 100.0, "timestamp": 1800010000,
    "city": "Tunis", "atm_id": "ATM_TUNIS_01"
})
affiche("TEST_03A (1ere TX)", r3a)

print()
print("  2eme TX 2min apres : ATM_TUNIS_03 (La Marsa, 15 km)")
r3b = post({
    "tx_id": "TEST_03B", "card_id": "CARTE_INTRA",
    "amount": 120.0, "timestamp": 1800010120,
    "city": "Tunis", "atm_id": "ATM_TUNIS_03"
})
affiche("TEST_03B (2eme TX)", r3b)

time.sleep(1)

# ============================================================
# ETAPE 5 : Nouvelle carte normale
# ============================================================
sep("ETAPE 5 : Nouvelle carte, transaction normale (ALLOW attendu)")
print("  Carte : CARTE_NOUVELLE | Ville : Sousse | 50 DT")
r4 = post({
    "tx_id": "TEST_04", "card_id": "CARTE_NOUVELLE",
    "amount": 50.0, "timestamp": 1800020000,
    "city": "Sousse", "atm_id": "ATM_SOUSSE_01"
})
affiche("TEST_04", r4)

# ============================================================
# ETAPE 6 : Stats Fog C
# ============================================================
sep("ETAPE 6 : Stats Fog C (decisions enregistrees)")
stats = get(f"{BASE_C}/stats")
if stats:
    print(f"  Total transactions  : {stats.get('total_transactions')}")
    print(f"  Bloquees            : {stats.get('blocked')}")
    print(f"  Autorisees          : {stats.get('allowed')}")
    print(f"  Taux de blocage     : {stats.get('block_rate_pct')} %")
    print(f"  Offline synchronise : {stats.get('offline_decisions_synced')}")
    print(f"  Raisons blocage     : {stats.get('blocked_by_reason')}")

# ============================================================
# ETAPE 7 : Liste des decisions
# ============================================================
sep("ETAPE 7 : Journal des decisions (Fog C)")
decisions = get(f"{BASE_C}/decisions?limit=20")
if decisions:
    print(f"  {'TX_ID':<12} {'CARTE':<16} {'VILLE':<10} {'MONTANT':>8}  {'DECISION':<8}  SOURCE")
    print("  " + "-" * 72)
    for d in reversed(decisions):
        print(f"  {d['tx_id']:<12} {d['card_id']:<16} {d['city']:<10} "
              f"{d['amount']:>8.1f}  {d['decision']:<8}  {d['source_node']}")

print()
print("=" * 60)
print("  TEST TERMINE")
print("  Pour tester la PANNE de Fog C :")
print("  1. Ferme le terminal Fog C (Ctrl+C)")
print("  2. Envoie une transaction -> fog_c_reachable=False")
print("  3. Relance Fog C -> sync automatique dans 30s")
print("=" * 60)
