# Architecture Distribuée Fog Computing pour la Détection de Fraude Bancaire (ATM)

Ce projet implémente un système multi-nœuds Fog coordonné sur 3 PC pour la détection en temps réel de fraudes sur distributeurs automatiques de billets (GAB).

---

## 📁 Arborescence du Projet

```text
frog/
│
├── fog_a/                         # 💻 PC 1 : Nœud Fog A (Analyse Comportementale IA)
│   ├── app.py                     # API FastAPI du nœud A (reçoit les retraits du GAB)
│   ├── inference.py               # Moteur d'inférence ONNX Runtime (< 1 ms)
│   ├── features.py                # Calcul des 9 variables comportementales
│   ├── config.json                # IP / Port du PC 2 (Fog B)
│   ├── requirements.txt           # Dépendances légères (numpy, onnxruntime, fastapi, uvicorn, httpx)
│   ├── README.md                  # Instructions de déploiement PC 1
│   └── artifacts/
│       ├── model.onnx             # Modèle IA Random Forest optimisé ONNX (2.1 Mo)
│       └── threshold.json         # Seuil optimal de détection (0.03)
│
├── fog_b/                         # 💻 PC 2 : Nœud Fog B (Analyse Spatio-Temporelle)
│   ├── app.py                     # API FastAPI du nœud B (vérification géographique)
│   ├── geo.py                     # Formule de Haversine & coordonnées des villes tunisiennes
│   ├── config.json                # IP / Port du PC 3 (Fog C) + Seuils de vitesse
│   ├── requirements.txt           # Dépendances (fastapi, uvicorn, httpx)
│   └── README.md                  # Instructions de déploiement PC 2
│
├── fog_c/                         # 💻 PC 3 : Nœud Fog C (Coordinateur & Décision Finale)
│   ├── app.py                     # API FastAPI du nœud C (décision ALLOW / BLOCK + audit)
│   ├── config.json                # Configuration et seuils de décision
│   ├── requirements.txt           # Dépendances (fastapi, uvicorn)
│   └── README.md                  # Instructions de déploiement PC 3
│
├── ml/                            # 🧪 Recherche & Entraînement Offline
│   ├── generate_dataset.py        # Simulateur ATM avec 4 types de fraudes
│   ├── features.py                # Définition des features comportementales et spatiales
│   ├── train.py                   # Entraînement du Random Forest & optimisation du coût
│   ├── evaluate.py                # Évaluation Fog A seul
│   ├── evaluate_pipeline.py       # Évaluation comparative Fog A vs Fog A + B + C
│   ├── export_onnx.py             # Export scikit-learn vers ONNX
│   ├── robustness.py              # Tests de robustesse et décalage de distribution
│   ├── test_replay.py             # Validation parité production vs offline
│   ├── requirements.txt           # Dépendances complètes d'entraînement
│   ├── data/
│   │   └── dataset_ATM.csv        # Dataset synthétique (~74 000 transactions)
│   └── artifacts/
│       ├── model.joblib           # Modèle scikit-learn brut
│       ├── model.onnx             # Modèle ONNX
│       ├── threshold.json         # Seuil validation
│       ├── metrics.json           # Métriques Fog A seul
│       └── metrics_pipeline.json  # Métriques combinées pipeline complet
│
└── demo_3pc.py                    # 🚀 Script de test local automatique (lance A, B, C en local)
```

---

## 🚀 Déploiement Physique sur 3 PC

1. **PC 1** : Copier le dossier `fog_a/`, installer `pip install -r requirements.txt`, renseigner l'IP du PC 2 dans `config.json`, lancer `python app.py`.
2. **PC 2** : Copier le dossier `fog_b/`, installer `pip install -r requirements.txt`, renseigner l'IP du PC 3 dans `config.json`, lancer `python app.py`.
3. **PC 3** : Copier le dossier `fog_c/`, installer `pip install -r requirements.txt`, lancer `python app.py`.

*Ordre de démarrage recommandé :* **PC 3** $\rightarrow$ **PC 2** $\rightarrow$ **PC 1**.

