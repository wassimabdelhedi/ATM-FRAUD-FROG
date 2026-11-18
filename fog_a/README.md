# Fog A - Analyse Comportementale IA (PC 1)

Noeud Fog A : point d'entree du pipeline de detection de fraude.
Analyse le comportement de la carte (rafales, montants anormaux, heures)
a l'aide d'un modele Random Forest exporte en ONNX.

## Fichiers

| Fichier | Role |
|---------|------|
| `app.py` | Serveur FastAPI (endpoint `/transaction`) |
| `inference.py` | Module d'inference ONNX + base SQLite historique |
| `features.py` | Calcul des 9 variables comportementales |
| `config.json` | Configuration reseau (IP de Fog B) |
| `artifacts/model.onnx` | Modele Random Forest (2.1 Mo) |
| `artifacts/threshold.json` | Seuil de decision (0.03) |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py --host 0.0.0.0 --port 8001
```

## Configuration

Modifier `config.json` avec l'adresse IP reelle du PC 2 (Fog B) :
```json
{
  "peers": {
    "fog_b_url": "http://<IP_PC_2>:8002"
  }
}
```

## Test rapide

```bash
curl -X POST http://localhost:8001/transaction -H "Content-Type: application/json" -d "{\"tx_id\":\"TX001\",\"card_id\":\"CARD001\",\"amount\":80,\"timestamp\":1790000000,\"city\":\"Tunis\"}"
```

