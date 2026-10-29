# ml/ — Modèle de Fog A (Random Forest -> ONNX)

    pip install -r requirements.txt
    python generate_dataset.py   # data/dataset_ATM.csv (schéma §10 du cahier des charges)
    python train.py              # artifacts/model.joblib  (split par carte)
    python evaluate.py           # artifacts/metrics.json + evaluation.png
    python export_onnx.py        # artifacts/model.onnx + vérification de parité
    python inference.py          # démo : BehaviorScorer

Fog A importe `features.py` (mêmes features qu'à l'entraînement) et `inference.py`.
Features V2 (9, ordre défini UNE fois dans `features.BEHAVIOR_FEATURES`) : amount, transactions_last_5min,
average_amount, amount_deviation, time_since_previous, transactions_last_30min, transactions_last_24h,
max_amount_7d, hour_of_day. distance/speed restent pour Fog B (règles), agrégats régionaux pour Fog C.
Changer la liste => regénérer, réentraîner, réexporter. Le .onnx contient la liste (métadonnée `features`)
et `BehaviorScorer` refuse un modèle incohérent avec `features.py`.
Hypothèses du simulateur pour hour_of_day : `--night-cards` (0.10) et `--night-fraud` (0.35).
Dataset SYNTHÉTIQUE : les résultats (notamment le rappel par type de fraude) dépendent de la simulation.
