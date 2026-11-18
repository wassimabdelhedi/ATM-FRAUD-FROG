# Fog C - Coordinateur & Decision Finale (PC 3)

Noeud Fog C : recoit les diagnostics combines de Fog A et Fog B,
rend la decision finale (ALLOWED / BLOCKED) et journalise toutes
les decisions dans une base SQLite pour l'audit.

## Fichiers

| Fichier | Role |
|---------|------|
| `app.py` | Serveur FastAPI (endpoint `/decide`) |
| `config.json` | Seuils de decision |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py --host 0.0.0.0 --port 8003
```

## Endpoints

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/health` | GET | Etat du noeud |
| `/decide` | POST | Decision finale (appele par Fog B) |
| `/decisions` | GET | Historique des decisions recentes |
| `/stats` | GET | Statistiques globales (taux de blocage) |

## Regles de decision

| Fog A (score >= 0.03) | Fog B (vitesse > 130 km/h) | Decision |
|:---:|:---:|:---:|
| Non | Non | ALLOWED |
| Oui | Non | BLOCKED (BEHAVIOR) |
| Non | Oui | BLOCKED (IMPOSSIBLE_TRAVEL) |
| Oui | Oui | BLOCKED (BEHAVIOR + TRAVEL) |

