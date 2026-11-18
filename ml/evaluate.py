"""Évaluation sur le TEST : métriques au seuil de validation, PR-AUC + IC bootstrap par carte,
métriques par épisode, 3 baselines, importance par permutation. Dataset synthétique."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import BEHAVIOR_FEATURES, FEATURES_VERSION
from train import COST_FP, MODEL, THRESHOLD_FILE, load_split3, pick_threshold, total_cost

FRAUD_TYPES = ["behavior", "card_testing", "impossible_travel", "distributed"]


def episode_metrics(test, flagged):
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


def bootstrap_ap(test, proba, n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = test.reset_index(drop=True).groupby("card_id").indices
    cards = np.array(list(idx))
    y = test["fraud"].to_numpy()
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[c] for c in rng.choice(cards, len(cards))])
        if y[take].sum():
            vals.append(average_precision_score(y[take], proba[take]))
    return [round(float(np.percentile(vals, q)), 4) for q in (2.5, 97.5)]


def summarize(name, test, flagged, score=None):
    y, f = test["fraud"].to_numpy(), np.asarray(flagged, dtype=bool)
    n_legit = int((y == 0).sum())
    out = {
        "precision": round(float(precision_score(y, f, zero_division=0)), 4),
        "recall": round(float(recall_score(y, f)), 4),
        "f1": round(float(f1_score(y, f)), 4),
        "false_positives_per_1000_legit": round(float(((y == 0) & f).sum() / n_legit * 1000), 2),
        "cost": round(total_cost(y, f, test["amount"]), 1),
    }
    if score is not None:
        out["pr_auc"] = round(float(average_precision_score(y, score)), 4)
        out["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
    out.update({k: v for k, v in episode_metrics(test, f).items() if k != "episode_recall_by_type"})
    return out


if __name__ == "__main__":
    train, val, test = load_split3()
    model = joblib.load(MODEL)
    thr = json.load(open(THRESHOLD_FILE))["threshold"]
    X, y = test[BEHAVIOR_FEATURES], test["fraud"].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    flag = proba >= thr

    results = {"features_version": FEATURES_VERSION, "threshold (choisi sur validation)": thr,
               "cost_fp": COST_FP, "test_prevalence": round(float(y.mean()), 5)}
    results["random_forest"] = summarize("rf", test, flag, proba)
    results["random_forest"]["pr_auc_ci95_card_bootstrap"] = bootstrap_ap(test, proba)
    results["random_forest"]["confusion_matrix [[TN,FP],[FN,TP]]"] = confusion_matrix(y, flag).tolist()
    results["random_forest"]["recall_by_fraud_type (Fog A seul, par transaction)"] = {
        t: round(float(flag[(test.fraud_type == t).to_numpy()].mean()), 4) for t in FRAUD_TYPES}
    results["random_forest"]["episode_recall_by_type"] = episode_metrics(test, flag)["episode_recall_by_type"]

    rule = ((test.amount_deviation > 2.0) | (test.transactions_last_5min >= 2)).to_numpy()
    results["baseline_rule (écart>2x moyenne OU >=2 tx en 5 min)"] = summarize("rule", test, rule)
    scores = {}
    for name, mdl in [("baseline_tree_depth3", DecisionTreeClassifier(max_depth=3, random_state=0)),
                      ("baseline_logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)))]:
        mdl.fit(train[BEHAVIOR_FEATURES], train["fraud"])
        t_b, _ = pick_threshold(val["fraud"], mdl.predict_proba(val[BEHAVIOR_FEATURES])[:, 1], val["amount"])
        s = mdl.predict_proba(X)[:, 1]
        scores[name] = s
        results[name] = summarize(name, test, s >= t_b, s)
        results[name]["threshold"] = t_b

    rng = np.random.default_rng(0)
    sel = np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], 20000, replace=False)])
    pi = permutation_importance(model, X.iloc[sel], y[sel], scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=-1)
    results["permutation_importance (baisse de PR-AUC)"] = {
        f: round(float(m), 4) for f, m in sorted(zip(BEHAVIOR_FEATURES, pi.importances_mean), key=lambda x: -x[1])}

    json.dump(results, open("artifacts/metrics.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, s in [("Random Forest", proba)] + [(k, v) for k, v in scores.items()]:
        p, r, _ = precision_recall_curve(y, s)
        ax[0].plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.2f})")
    ax[0].axhline(y.mean(), ls="--", c="gray", label="hasard")
    ax[0].set_xlabel("rappel"); ax[0].set_ylabel("précision"); ax[0].set_title("Courbes PR (test)"); ax[0].legend(fontsize=7)
    cm = confusion_matrix(y, flag)
    ax[1].imshow(np.log1p(cm), cmap="Blues"); ax[1].set_title(f"Matrice de confusion (seuil {thr:.2f})")
    ax[1].set_xticks([0, 1], ["ALLOW", "BLOCK"]); ax[1].set_yticks([0, 1], ["légitime", "fraude"])
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i, j], ha="center", va="center", color="red")
    imp = results["permutation_importance (baisse de PR-AUC)"]
    ax[2].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[2].set_title("Importance par permutation")
    plt.tight_layout(); plt.savefig("artifacts/evaluation.png", dpi=130)