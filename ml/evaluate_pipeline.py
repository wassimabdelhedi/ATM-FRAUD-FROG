"""Evaluation combinee Fog A + Fog B sur le jeu de test complet.

Rejoue les transactions du test en simulant le pipeline :
  1. Fog A : scoring comportemental (ONNX, seuil 0.03)
  2. Fog B : verification spatio-temporelle (speed > 130 km/h)
  3. Fog C : decision finale (Fog A OR Fog B)

Compare les metriques Fog A seul vs Fog A + Fog B + Fog C.

Lancer :  python evaluate_pipeline.py
"""
import json
import time

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from features import BEHAVIOR_FEATURES, FEATURES_VERSION
from train import COST_FP, load_split3, total_cost

# ---- Seuils ----
with open("config.json") as f:
    CONFIG = json.load(f)

BEHAVIOR_THRESHOLD = CONFIG["thresholds"]["behavior_score"]
MAX_SPEED_KMH = CONFIG["thresholds"]["max_speed_kmh"]
MIN_DISTANCE_KM = CONFIG["thresholds"]["min_distance_km"]

FRAUD_TYPES = ["behavior", "card_testing", "impossible_travel", "distributed"]


def episode_metrics(test, flagged):
    """Calcule les metriques par episode de fraude."""
    d = test.assign(flag=np.asarray(flagged, dtype=bool))
    d = d[d["fraud_episode_id"] != ""].sort_values("transaction_id")
    detected, saved, total, delays, by_kind = 0, 0.0, 0.0, [], {}
    for _, g in d.groupby("fraud_episode_id"):
        g = g.reset_index(drop=True)
        total += g.amount.sum()
        hit = g.index[g.flag]
        kind = g.fraud_type.iloc[0]
        by_kind.setdefault(kind, [0, 0])[1] += 1
        if len(hit):
            k = int(hit[0])
            detected += 1
            saved += g.amount.iloc[k:].sum()
            delays.append(k)
            by_kind[kind][0] += 1
    n = d["fraud_episode_id"].nunique()
    return {
        "episodes": int(n),
        "episode_recall": round(detected / n, 4) if n else None,
        "amount_saved_rate": round(float(saved / total), 4) if total else None,
        "median_fraud_tx_before_first_alert": float(np.median(delays)) if delays else None,
        "episode_recall_by_type": {k: round(v[0] / v[1], 4) for k, v in by_kind.items()},
    }


def summarize(y, flagged, amounts):
    """Resume les metriques de classification."""
    f = np.asarray(flagged, dtype=bool)
    n_legit = int((y == 0).sum())
    return {
        "precision": round(float(precision_score(y, f, zero_division=0)), 4),
        "recall": round(float(recall_score(y, f)), 4),
        "f1": round(float(f1_score(y, f)), 4),
        "false_positives_per_1000_legit": round(float(((y == 0) & f).sum() / n_legit * 1000), 2),
        "cost": round(total_cost(y, f, amounts), 1),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  EVALUATION COMBINEE  Fog A + Fog B + Fog C")
    print("=" * 70)

    # Charger les donnees
    _, _, test = load_split3()
    model = joblib.load("artifacts/model.joblib")
    thr = json.load(open("artifacts/threshold.json"))["threshold"]

    X = test[BEHAVIOR_FEATURES]
    y = test["fraud"].to_numpy()
    amounts = test["amount"].to_numpy()

    print(f"\nDataset test : {len(test)} transactions | "
          f"{int(y.sum())} fraudes | prevalence = {y.mean():.4%}")
    print(f"Seuil comportemental (Fog A) : {thr}")
    print(f"Seuil vitesse max    (Fog B) : {MAX_SPEED_KMH} km/h")
    print(f"Seuil distance min   (Fog B) : {MIN_DISTANCE_KM} km")

    # ---- Fog A seul ----
    print("\n" + "-" * 70)
    print("  FOG A SEUL (Analyse Comportementale IA)")
    print("-" * 70)
    proba = model.predict_proba(X)[:, 1]
    flag_a = proba >= thr

    m_a = summarize(y, flag_a, amounts)
    ep_a = episode_metrics(test, flag_a)
    cm_a = confusion_matrix(y, flag_a)

    print(f"  Precision      : {m_a['precision']:.4f}")
    print(f"  Rappel         : {m_a['recall']:.4f}")
    print(f"  F1             : {m_a['f1']:.4f}")
    print(f"  FP / 1000 leg. : {m_a['false_positives_per_1000_legit']}")
    print(f"  Cout total     : {m_a['cost']:.0f}")
    print(f"  Episode recall : {ep_a['episode_recall']:.4f}")
    print(f"  Montant sauve  : {ep_a['amount_saved_rate']:.4f}")
    print(f"  Matrice [[TN,FP],[FN,TP]] : {cm_a.tolist()}")

    print("  Rappel par type (Fog A seul) :")
    for t in FRAUD_TYPES:
        mask = (test.fraud_type == t).to_numpy()
        if mask.sum() > 0:
            r = flag_a[mask].mean()
            print(f"    {t:20s} : {r:.4f} ({int(flag_a[mask].sum())}/{mask.sum()})")

    # ---- Fog B : utilisation directe des colonnes speed et distance ----
    print("\n" + "-" * 70)
    print("  FOG B (Analyse Spatio-Temporelle)")
    print("-" * 70)

    # Les colonnes speed et distance_from_previous_atm sont deja calculees
    speed = test["speed"].to_numpy()
    distance = test["distance_from_previous_atm"].to_numpy()

    flag_b = (speed > MAX_SPEED_KMH) & (distance > MIN_DISTANCE_KM)

    n_b_alerts = flag_b.sum()
    n_b_tp = (flag_b & (y == 1)).sum()
    n_b_fp = (flag_b & (y == 0)).sum()
    print(f"  Alertes Fog B totales       : {n_b_alerts}")
    print(f"    dont vraies fraudes (TP)   : {n_b_tp}")
    print(f"    dont faux positifs  (FP)   : {n_b_fp}")

    # Rappel Fog B par type
    print("  Rappel Fog B par type :")
    for t in FRAUD_TYPES:
        mask = (test.fraud_type == t).to_numpy()
        if mask.sum() > 0:
            r = flag_b[mask].mean()
            print(f"    {t:20s} : {r:.4f} ({int(flag_b[mask].sum())}/{mask.sum()})")

    # ---- Fog C : Decision Finale (Fog A OU Fog B) ----
    print("\n" + "-" * 70)
    print("  FOG C : DECISION FINALE (Fog A + Fog B combines)")
    print("-" * 70)

    flag_combined = flag_a | flag_b

    m_c = summarize(y, flag_combined, amounts)
    ep_c = episode_metrics(test, flag_combined)
    cm_c = confusion_matrix(y, flag_combined)

    print(f"  Precision      : {m_c['precision']:.4f}")
    print(f"  Rappel         : {m_c['recall']:.4f}")
    print(f"  F1             : {m_c['f1']:.4f}")
    print(f"  FP / 1000 leg. : {m_c['false_positives_per_1000_legit']}")
    print(f"  Cout total     : {m_c['cost']:.0f}")
    print(f"  Episode recall : {ep_c['episode_recall']:.4f}")
    print(f"  Montant sauve  : {ep_c['amount_saved_rate']:.4f}")
    print(f"  Matrice [[TN,FP],[FN,TP]] : {cm_c.tolist()}")

    print("  Rappel par type (Fog A + Fog B) :")
    for t in FRAUD_TYPES:
        mask = (test.fraud_type == t).to_numpy()
        if mask.sum() > 0:
            r = flag_combined[mask].mean()
            print(f"    {t:20s} : {r:.4f} ({int(flag_combined[mask].sum())}/{mask.sum()})")

    # ---- Comparaison ----
    print("\n" + "=" * 70)
    print("  COMPARAISON : Fog A seul vs Fog A + Fog B + Fog C")
    print("=" * 70)
    print(f"  {'Metrique':<30s} {'Fog A seul':>12s} {'A + B + C':>12s} {'Delta':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    comparisons = [
        ("Rappel", m_a["recall"], m_c["recall"]),
        ("Precision", m_a["precision"], m_c["precision"]),
        ("F1", m_a["f1"], m_c["f1"]),
        ("FP / 1000 legit", m_a["false_positives_per_1000_legit"],
         m_c["false_positives_per_1000_legit"]),
        ("Cout total", m_a["cost"], m_c["cost"]),
        ("Episode recall", ep_a["episode_recall"], ep_c["episode_recall"]),
        ("Montant sauve %", ep_a["amount_saved_rate"], ep_c["amount_saved_rate"]),
    ]
    for name, va, vc in comparisons:
        delta = vc - va
        sign = "+" if delta > 0 else ""
        print(f"  {name:<30s} {va:>12.4f} {vc:>12.4f} {sign}{delta:>11.4f}")

    print(f"\n  {'Type de fraude':<30s} {'Fog A seul':>12s} {'A + B + C':>12s} {'Delta':>10s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    for t in FRAUD_TYPES:
        mask = (test.fraud_type == t).to_numpy()
        if mask.sum() > 0:
            ra = flag_a[mask].mean()
            rc = flag_combined[mask].mean()
            delta = rc - ra
            sign = "+" if delta > 0 else ""
            print(f"  {t:<30s} {ra:>12.4f} {rc:>12.4f} {sign}{delta:>9.4f}")

    # ---- Sauvegarder ----
    results = {
        "features_version": FEATURES_VERSION,
        "thresholds": CONFIG["thresholds"],
        "fog_a_only": {**m_a, **{k: v for k, v in ep_a.items() if k != "episode_recall_by_type"},
                       "confusion_matrix": cm_a.tolist(),
                       "recall_by_type": {t: round(float(flag_a[(test.fraud_type == t).to_numpy()].mean()), 4)
                                          for t in FRAUD_TYPES if (test.fraud_type == t).sum() > 0},
                       "episode_recall_by_type": ep_a.get("episode_recall_by_type", {})},
        "fog_abc_combined": {**m_c, **{k: v for k, v in ep_c.items() if k != "episode_recall_by_type"},
                             "confusion_matrix": cm_c.tolist(),
                             "recall_by_type": {t: round(float(flag_combined[(test.fraud_type == t).to_numpy()].mean()), 4)
                                                for t in FRAUD_TYPES if (test.fraud_type == t).sum() > 0},
                             "episode_recall_by_type": ep_c.get("episode_recall_by_type", {})},
        "fog_b_standalone": {
            "total_alerts": int(n_b_alerts),
            "true_positives": int(n_b_tp),
            "false_positives": int(n_b_fp),
        },
    }

    out_path = "artifacts/metrics_pipeline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultats sauvegardes -> {out_path}")
    print("\n[SUCCESS] Evaluation terminee !")
