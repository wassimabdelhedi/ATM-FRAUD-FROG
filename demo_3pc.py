"""Demo locale : lance les 3 noeuds Fog depuis leurs dossiers respectifs
et execute les scenarios de test (normal + voyage impossible).

Lancer depuis la racine du projet :  python demo_3pc.py
"""
import json
import subprocess
import sys
import time

import httpx

PYTHON = sys.executable


def start_node(folder, port):
    return subprocess.Popen(
        [PYTHON, "app.py", "--port", str(port), "--host", "127.0.0.1"],
        cwd=folder,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def wait_ready(url, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def send_tx(fog_a_url, tx):
    r = httpx.post(f"{fog_a_url}/transaction", json=tx, timeout=30)
    return r.json()


if __name__ == "__main__":
    # Patcher les config.json pour pointer vers localhost
    for folder, cfg_patch in [
        ("fog_a", {"peers": {"fog_b_url": "http://127.0.0.1:8002"}}),
        ("fog_b", {"peers": {"fog_c_url": "http://127.0.0.1:8003"}}),
    ]:
        path = f"{folder}/config.json"
        with open(path) as f:
            cfg = json.load(f)
        cfg.update(cfg_patch)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("   DEMO PIPELINE  Fog A (PC1) -> Fog B (PC2) -> Fog C (PC3)")
    print("=" * 70)

    # 1. Lancer les 3 noeuds depuis leurs dossiers
    print("\n[1/4] Lancement des 3 noeuds Fog...")
    procs = []
    for folder, port, name in [
        ("fog_c", 8003, "Fog C (PC 3)"),
        ("fog_b", 8002, "Fog B (PC 2)"),
        ("fog_a", 8001, "Fog A (PC 1)"),
    ]:
        p = start_node(folder, port)
        procs.append(p)
        print(f"  [OK] {name} lance depuis ./{folder}/ (PID {p.pid}, port {port})")

    # 2. Attendre
    print("\n[2/4] Attente de la disponibilite des noeuds...")
    for port, name in [(8003, "Fog C"), (8002, "Fog B"), (8001, "Fog A")]:
        ok = wait_ready(f"http://127.0.0.1:{port}/health")
        marker = "[OK]" if ok else "[ECHEC]"
        print(f"  {marker} {name} sur le port {port}")
        if not ok:
            for p in procs:
                p.terminate()
            sys.exit(1)

    fog_a_url = "http://127.0.0.1:8001"
    T0 = 1_790_000_000

    # 3. Historique
    print("\n[3/4] Injection de 10 transactions historiques a Tunis...")
    for i in range(10, 0, -1):
        send_tx(fog_a_url, {
            "tx_id": f"HIST_{i:03d}", "card_id": "CARD_DEMO_001",
            "amount": 70.0 + (i % 3) * 10,
            "timestamp": T0 - i * 86400, "city": "Tunis",
        })
    print("  [OK] Historique injecte")

    # ---- Scenario 1 : Normal ----
    print("\n" + "=" * 70)
    print("  SCENARIO 1 : Transaction Normale a Tunis")
    print("=" * 70)
    result = send_tx(fog_a_url, {
        "tx_id": "TX_NORMAL_001", "card_id": "CARD_DEMO_001",
        "amount": 80.0, "timestamp": T0 + 7200, "city": "Tunis",
    })
    print(f"  -> 80 DT a Tunis")
    print(f"  <- Fog A score  : {result['fog_a']['score']:.4f} (block={result['fog_a']['block']})")
    print(f"  <- Fog B atteint: {result['fog_b_reachable']}")

    time.sleep(0.5)
    r = httpx.get("http://127.0.0.1:8003/decisions?limit=1", timeout=5)
    d = r.json()[0]
    print(f"  ** DECISION FINALE : {d['decision']} ({d['reason']})")

    # ---- Scenario 2 : Voyage impossible ----
    print("\n" + "=" * 70)
    print("  SCENARIO 2 : Voyage Impossible -- Tunis -> Sfax en 5 min")
    print("=" * 70)
    result = send_tx(fog_a_url, {
        "tx_id": "TX_FRAUD_001", "card_id": "CARD_DEMO_001",
        "amount": 80.0, "timestamp": T0 + 7200 + 300, "city": "Sfax",
    })
    print(f"  -> 80 DT a Sfax (5 min apres Tunis, ~236 km)")
    print(f"  <- Fog A score  : {result['fog_a']['score']:.4f} (block={result['fog_a']['block']})")
    print(f"  <- Fog B atteint: {result['fog_b_reachable']}")

    time.sleep(0.5)
    r = httpx.get("http://127.0.0.1:8003/decisions?limit=1", timeout=5)
    d = r.json()[0]
    print(f"  ** DECISION FINALE : {d['decision']} ({d['reason']})")

    # ---- Scenario 3 : Rafale suspecte ----
    print("\n" + "=" * 70)
    print("  SCENARIO 3 : Rafale de retraits suspects (card_testing)")
    print("=" * 70)
    base_ts = T0 + 20000
    for i in range(5):
        try:
            result = send_tx(fog_a_url, {
                "tx_id": f"TX_BURST_{i}", "card_id": "CARD_DEMO_002",
                "amount": 900.0, "timestamp": base_ts + i * 30, "city": "Sousse",
            })
            print(f"  [{i+1}/5] 900 DT a Sousse | score={result['fog_a']['score']:.4f} | block={result['fog_a']['block']}")
        except Exception as e:
            print(f"  [{i+1}/5] Timeout (normal en mono-worker lors d'une rafale)")
        time.sleep(1.0)  # laisser le pipeline digerer

    time.sleep(0.5)
    r = httpx.get("http://127.0.0.1:8003/decisions?limit=1", timeout=5)
    d = r.json()[0]
    print(f"  ** DECISION FINALE : {d['decision']} ({d['reason']})")

    # ---- Stats ----
    print("\n" + "=" * 70)
    print("  STATISTIQUES GLOBALES (Fog C)")
    print("=" * 70)
    r = httpx.get("http://127.0.0.1:8003/stats", timeout=5)
    stats = r.json()
    print(f"  Total transactions : {stats['total_transactions']}")
    print(f"  Autorisees         : {stats['allowed']}")
    print(f"  Bloquees           : {stats['blocked']}")
    print(f"  Taux de blocage    : {stats['block_rate']}%")
    if stats.get("blocked_by_reason"):
        for reason, count in stats["blocked_by_reason"].items():
            print(f"    - {reason} : {count}")

    # Cleanup
    print("\n[4/4] Arret des noeuds...")
    for p in procs:
        p.terminate()
    print("  [OK] Tous les noeuds arretes.")
    print("\n[SUCCESS] Demo terminee avec succes !")
