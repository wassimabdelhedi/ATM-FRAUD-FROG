# Fog B - Analyse Spatio-Temporelle (PC 2)

Noeud Fog B : verifie la coherence geographique des transactions.
Detecte les voyages impossibles (carte clonee) en calculant la vitesse
entre deux retraits dans des villes differentes.

## Fichiers

| Fichier | Role |
|---------|------|
| `app.py` | Serveur FastAPI (endpoint `/analyze`) |
| `geo.py` | Coordonnees des villes + calcul Haversine |
| `config.json` | Configuration reseau (IP de Fog C) + seuils |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py --host 0.0.0.0 --port 8002
```

## Configuration

Modifier `config.json` avec l'adresse IP reelle du PC 3 (Fog C) :
```json
{
  "peers": {
    "fog_c_url": "http://<IP_PC_3>:8003"
  },
  "thresholds": {
    "max_speed_kmh": 130.0,
    "min_distance_km": 20.0
  }
}
```

## Seuils

- **Vitesse max** : 130 km/h (au-dela = voyage impossible)
- **Distance min** : 20 km (en dessous = meme zone, pas d'alerte)

