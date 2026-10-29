"""Évaluation : Accuracy, Precision, Recall, F1, ROC-AUC, matrice de confusion + graphiques.
Ajoute le rappel PAR TYPE de fraude : montre ce que Fog A voit seul (base de l'expérience 3)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from features import BEHAVIOR_FEATURES, FEATURES_VERSION
from train import MODEL, load_split

THRESHOLD = 0.6

if __name__ == "__main__":
    _, test = load_split()
    model = joblib.load(MODEL)
    X, y = test[BEHAVIOR_FEATURES], test["fraud"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    by_type = {t: round(float(pred[(test.fraud_type == t).values].mean()), 4)
               for t in ["behavior", "impossible_travel", "distributed"]}
    metrics = {
        "features_version": FEATURES_VERSION,
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y, pred).tolist(),
        "recall_by_fraud_type (Fog A seul)": by_type,
        "feature_importance": dict(zip(BEHAVIOR_FEATURES, map(lambda v: round(float(v), 4), model.feature_importances_))),
    }
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    json.dump(metrics, open("artifacts/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cm = confusion_matrix(y, pred)
    ax[0].imshow(cm, cmap="Blues"); ax[0].set_title("Matrice de confusion")
    ax[0].set_xticks([0, 1], ["ALLOW", "BLOCK"]); ax[0].set_yticks([0, 1], ["légitime", "fraude"])
    for (i, j), v in ((ij, cm[ij]) for ij in [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax[0].text(j, i, v, ha="center", va="center", color="red")
    fpr, tpr, _ = roc_curve(y, proba)
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "--", c="gray")
    ax[1].set_title(f"ROC (AUC={metrics['roc_auc']})"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[2].barh(BEHAVIOR_FEATURES, model.feature_importances_); ax[2].set_title("Importance des features")
    plt.tight_layout(); plt.savefig("artifacts/evaluation.png", dpi=130)
