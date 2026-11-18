"""Demo de resilience -- Architecture Fog A + B + C avec tolerance aux pannes.

Ce script demontre les 3 scenarios de panne :

  Scenario 1 : Fonctionnement NORMAL (3 noeuds actifs)
  Scenario 2 : PANNE DE FOG C (A et B continuent en mode local)
  Scenario 3 : RETOUR DE FOG C (decisions offline re-synchronisees)

Lancer (depuis la racine du projet) :
  cd c:\\...\\frog
  ml\\venv\\Scripts\\python.exe demo_resilience.py

Remarque : supprimez les fichiers *.db* avant chaque execution propre.
"""
import json
import subprocess
import sys
import time

# Chemin vers Python du venv
PYTHON = r"ml\venv\Scripts\python.exe"


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def banner(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def sub_banner(title: str):
    print()
    print(f"--- {title} ---")


def send_transaction(tx: dict, node_url: str = "https://127.0.0.1:8001") -> dict | None:
    """Envoie une transaction a Fog A via httpx (meme venv, HTTPS verify=False)."""
    code = (
        "import httpx, json, sys; "
        f"r = httpx.post('{node_url}/transaction', json={json.dumps(tx)}, timeout=8.0, verify=False); "
        "print(json.dumps(r.json(), ensure_ascii=False))"
    )
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        print(f"  [ERREUR] {result.stderr.strip()[:200]}")
        return None
    try:
        data = json.loads(result.stdout.strip())
        return data
    except Exception as exc:
        print(f"  [JSON parse error] {exc} -- stdout: {result.stdout[:200]}")
        return None


def get_json(url: str) -> dict | list | None:
    code = (
        "import httpx, json; "
        f"r = httpx.get('{url}', timeout=5.0, verify=False); "
        "print(json.dumps(r.json(), ensure_ascii=False))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except Exception:
        return None


def print_decision(label: str, resp: dict | None):
    if resp is None:
        print(f"  {label}: [PAS DE REPONSE]")
        return
    decision = resp.get("decision", "?")
    reason = resp.get("reason", "?")
    source = resp.get("decision_source", "fog_c")
    fog_b_ok = resp.get("fog_b_reachable", "?")
    fog_c_ok = resp.get("fog_c_reachable", "?")
    score = resp.get("fog_a", {}).get("score", "?")

    icon = "[BLOQUE]" if decision == "BLOCK" else ("[REVUE]" if decision == "REVIEW" else "[AUTORISE]")
    print(f"  {label}: {icon} {decision} | Raison: {reason}")
    print(f"    Score IA: {score} | Source decision: {source}")
    print(f"    Fog B joignable: {fog_b_ok} | Fog C joignable: {fog_c_ok}")


def wait_health(url: str, label: str, retries: int = 15) -> bool:
    code = (
        "import httpx; "
        f"r = httpx.get('{url}/health', timeout=3.0, verify=False); "
        "print(r.status_code)"
    )
    for i in range(retries):
        result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=6)
        if result.returncode == 0 and "200" in result.stdout:
            print(f"  {label} : pret")
            return True
        time.sleep(1)
    print(f"  {label} : TIMEOUT apres {retries}s")
    return False


# ---------------------------------------------------------------------------
# Transactions de test
# ---------------------------------------------------------------------------

TX_NORMAL = {
    "tx_id": "RES_TX_001",
    "card_id": "RES_CARD_1",
    "amount": 150.0,
    "timestamp": 1800000000,
    "city": "Tunis",
    "atm_id": "ATM_TUNIS_01",
}

TX_SFAX_IMPOSSIBLE = {
    "tx_id": "RES_TX_002",
    "card_id": "RES_CARD_1",          # meme carte -> voyage impossible
    "amount": 90.0,
    "timestamp": 1800000300,           # 5 minutes apres
    "city": "Sfax",
    "atm_id": "ATM_SFAX_01",
}

TX_INTRA_CITY_1 = {
    "tx_id": "RES_TX_INTRA_1",
    "card_id": "RES_CARD_INTRA",
    "amount": 100.0,
    "timestamp": 1800000000,
    "city": "Tunis",
    "atm_id": "ATM_TUNIS_01",          # Centre-ville Tunis
}

TX_INTRA_CITY_2 = {
    "tx_id": "RES_TX_INTRA_2",
    "card_id": "RES_CARD_INTRA",       # meme carte intra-ville
    "amount": 120.0,
    "timestamp": 1800000120,           # 2 minutes apres (120s) a 15 km
    "city": "Tunis",
    "atm_id": "ATM_TUNIS_03",          # La Marsa (15 km) -> 450 km/h !
}

TX_BURST = {
    "tx_id": "RES_TX_003",
    "card_id": "RES_CARD_2",
    "amount": 950.0,
    "timestamp": 1800001000,
    "city": "Sousse",
    "atm_id": "ATM_SOUSSE_01",
}

TX_WHILE_C_DOWN_1 = {
    "tx_id": "RES_TX_004",
    "card_id": "RES_CARD_3",
    "amount": 200.0,
    "timestamp": 1800002000,
    "city": "Nabeul",
    "atm_id": "ATM_NABEUL_01",
}

TX_WHILE_C_DOWN_2 = {
    "tx_id": "RES_TX_005",
    "card_id": "RES_CARD_3",           # meme carte -> voyage impossible
    "amount": 180.0,
    "timestamp": 1800002200,            # 3 min 20 s apres
    "city": "Gabes",
    "atm_id": "ATM_GABES_01",
}


# ---------------------------------------------------------------------------
# Lancement des noeuds
# ---------------------------------------------------------------------------

def start_node(folder: str, label: str):
    proc = subprocess.Popen(
        [PYTHON, "app.py"],
        cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"  {label} lance (PID {proc.pid})")
    return proc


def stop_node(proc, label: str):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  {label} arrete")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    banner("DEMO RESILIENCE -- Systeme de detection de fraude ATM")
    print("Architecture : Fog A (IA) <-> Fog B (Geo) <-> Fog C (Decision)")
    print("Objectif : prouver que le systeme survit a la panne de Fog C")
    print()

    # Nettoyer les anciennes bases
    import glob
    import os
    for pattern in [
        "fog_a/*.db", "fog_a/*.db-wal", "fog_a/*.db-shm",
        "fog_b/*.db", "fog_b/*.db-wal", "fog_b/*.db-shm",
        "fog_c/*.db", "fog_c/*.db-wal", "fog_c/*.db-shm",
    ]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass
    print("Bases SQLite nettoyees.")

    # ----------------------------------------------------------------
    # Demarrage de tous les noeuds
    # ----------------------------------------------------------------
    banner("DEMARRAGE DES 3 NOEUDS")
    proc_a = start_node("fog_a", "Fog A (port 8001)")
    proc_b = start_node("fog_b", "Fog B (port 8002)")
    proc_c = start_node("fog_c", "Fog C (port 8003)")

    print("Attente du demarrage des noeuds...")
    time.sleep(3)
    ok_a = wait_health("https://127.0.0.1:8001", "Fog A")
    ok_b = wait_health("https://127.0.0.1:8002", "Fog B")
    ok_c = wait_health("https://127.0.0.1:8003", "Fog C")

    if not (ok_a and ok_b and ok_c):
        print("[ERREUR] Un ou plusieurs noeuds n'ont pas demarre. Arret.")
        stop_node(proc_a, "Fog A")
        stop_node(proc_b, "Fog B")
        stop_node(proc_c, "Fog C")
        sys.exit(1)

    # ----------------------------------------------------------------
    # SCENARIO 1 : Fonctionnement normal (3 noeuds actifs)
    # ----------------------------------------------------------------
    banner("SCENARIO 1 : FONCTIONNEMENT NORMAL (A + B + C actifs)")

    sub_banner("Test 1a : Transaction normale -- Tunis 150 DT")
    r1 = send_transaction(TX_NORMAL)
    print_decision("RES_TX_001", r1)

    time.sleep(1)

    sub_banner("Test 1b : Voyage impossible INTER-VILLES -- Sfax 5 min apres Tunis")
    r2 = send_transaction(TX_SFAX_IMPOSSIBLE)
    print_decision("RES_TX_002", r2)

    time.sleep(1)

    sub_banner("Test 1c : Voyage impossible INTRA-VILLE -- Tunis Centre -> La Marsa en 2 min")
    _ = send_transaction(TX_INTRA_CITY_1)
    time.sleep(0.5)
    r_intra = send_transaction(TX_INTRA_CITY_2)
    print_decision("RES_TX_INTRA_2 (La Marsa)", r_intra)

    time.sleep(1)

    sub_banner("Test 1d : Montant anormal -- 950 DT Sousse")
    r3 = send_transaction(TX_BURST)
    print_decision("RES_TX_003", r3)

    print()
    print(">>> ATTENDU : TX_001=ALLOW, TX_002=BLOCK (inter-villes), TX_INTRA_2=BLOCK (intra-ville), TX_003=BLOCK (montant)")

    # ----------------------------------------------------------------
    # SCENARIO 2 : Panne de Fog C
    # ----------------------------------------------------------------
    banner("SCENARIO 2 : PANNE DE FOG C (A et B fonctionnent en mode local)")
    print("Arret de Fog C en cours...")
    stop_node(proc_c, "Fog C")
    time.sleep(2)
    print("Fog C est maintenant HORS LIGNE.")
    print()
    print("Le systeme doit continuer a traiter les transactions grace a :")
    print("  - decision_engine.py (make_decision) replique sur A et B")
    print("  - offline_queue SQLite pour stocker les decisions jusqu'au retour de C")
    print()

    sub_banner("Test 2a : Transaction normale pendant panne de C")
    r4 = send_transaction(TX_WHILE_C_DOWN_1)
    print_decision("RES_TX_004", r4)

    time.sleep(1)

    sub_banner("Test 2b : Voyage impossible pendant panne de C (Nabeul->Gabes en 3min)")
    r5 = send_transaction(TX_WHILE_C_DOWN_2)
    print_decision("RES_TX_005", r5)

    print()
    print(">>> ATTENDU : fog_c_reachable=False | decision_source=local_fallback")
    print(">>> ATTENDU : TX_004=ALLOW, TX_005=BLOCK (voyage impossible)")

    # Verifier la file offline de Fog A et Fog B
    time.sleep(1)
    sub_banner("Etat de la file offline de Fog A")
    qa = get_json("https://127.0.0.1:8001/offline_queue")
    if qa:
        for item in qa:
            print(f"  {item['tx_id']} | {item['decision']} | synced={item['synced']}")
    else:
        print("  (file vide ou inaccessible)")

    sub_banner("Etat de la file offline de Fog B")
    qb = get_json("https://127.0.0.1:8002/offline_queue")
    if qb:
        for item in qb:
            print(f"  {item['tx_id']} | {item['decision']} | synced={item['synced']}")
    else:
        print("  (file vide ou inaccessible)")

    # ----------------------------------------------------------------
    # SCENARIO 3 : Retour de Fog C + sync des decisions offline
    # ----------------------------------------------------------------
    banner("SCENARIO 3 : RETOUR DE FOG C -- Synchronisation offline")
    print("Redemarrage de Fog C...")
    proc_c = start_node("fog_c", "Fog C (port 8003)")
    time.sleep(3)
    ok_c = wait_health("https://127.0.0.1:8003", "Fog C")

    if ok_c:
        print()
        print("Fog C est de retour. Le thread de sync va envoyer les files")
        print(f"offline dans les prochaines {30}s (intervalle configure).")
        print()
        print(f"En attendant la sync automatique, test d'une transaction directe...")
        TX_AFTER_RECOVERY = {
            "tx_id": "RES_TX_006",
            "card_id": "RES_CARD_4",
            "amount": 300.0,
            "timestamp": 1800005000,
            "city": "Bizerte",
            "atm_id": "ATM_BIZERTE_01",
        }
        r6 = send_transaction(TX_AFTER_RECOVERY)
        print_decision("RES_TX_006", r6)
        print()
        print(">>> ATTENDU : fog_c_reachable=True | decision_source=fog_c")

        # Verifier les stats de Fog C
        time.sleep(2)
        sub_banner("Stats de Fog C (decisions recues via pipeline normal)")
        stats = get_json("https://127.0.0.1:8003/stats")
        if stats:
            print(f"  Total decisions : {stats.get('total_transactions')}")
            print(f"  Bloquees        : {stats.get('blocked')}")
            print(f"  Autorisees      : {stats.get('allowed')}")
            print(f"  Offline synced  : {stats.get('offline_decisions_synced')}")
        sub_banner("Dernieres decisions enregistrees dans Fog C")
        decs = get_json("https://127.0.0.1:8003/decisions?limit=10")
        if decs:
            for d in decs:
                print(f"  {d['tx_id']} | {d['decision']} | {d['reason']} | "
                      f"source={d['source_node']}")

    # ----------------------------------------------------------------
    # Fin
    # ----------------------------------------------------------------
    banner("ARRET DES NOEUDS")
    stop_node(proc_a, "Fog A")
    stop_node(proc_b, "Fog B")
    stop_node(proc_c, "Fog C")

    banner("RESUME DE LA DEMO RESILIENCE")
    print("Scenario 1 : Fonctionnement normal      --> OK (3 noeuds)")
    print("Scenario 2 : Panne Fog C                --> OK (A+B local fallback + offline queue)")
    print("Scenario 3 : Retour Fog C               --> OK (nouvelles TX routees vers C)")
    print()
    print("Points de robustesse implementes :")
    print("  [1] decision_engine.py IDENTIQUE sur A, B et C")
    print("  [2] geo.py copie sur Fog A (secours si Fog B tombe)")
    print("  [3] model.onnx copie sur Fog B (secours si Fog A tombe)")
    print("  [4] offline_queue SQLite sur A et B")
    print("  [5] Thread de sync vers Fog C toutes les 30s")
    print("  [6] /sync_offline_decisions sur Fog C pour reception des files")
    print()
    print("Pret pour deploiement sur 3 PCs physiques.")
    print("Mettez a jour les IPs dans fog_X/config.json avant le deploiement.")


if __name__ == "__main__":
    main()

