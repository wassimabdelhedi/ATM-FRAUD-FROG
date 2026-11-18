"""Fonctions geographiques et referentiel des GABs (Automates).

Ce module contient :
  1. Les coordonnees des centres-villes tunisiens (CITIES).
  2. Le referentiel simule des GABs par ville (ATMS) : 3 automates par ville
     decales de quelques kilometres (centre, banlieue/quartier nord, quartier sud).
  3. Le calcul de distance geodesique de Haversine (haversine_km).
  4. La fonction get_coordinates() qui resout d'abord l'atm_id precis,
     puis se replie sur le centre de la ville si l'atm_id est inconnu.

NOTE METHODOLOGIQUE (Rapport / Memoire) :
  Les banques tunisiennes ne diffusent pas publiquement les coordonnees
  GPS exactes de leurs distributeurs pour des raisons de securite
  (securite physique des transferts de fonds). Ces coordonnees sont donc
  simulees de facon realiste autour des centres urbains.
  Si une banque ou un export OpenStreetMap (amenity=atm) fournit
  les vrais emplacements, il suffit de remplacer le dictionnaire ATMS
  sans modifier aucune ligne du code algorithmique.
"""
import math

# ---------------------------------------------------------------------------
# 1. Coordonnees des centres-villes tunisiens
# ---------------------------------------------------------------------------
CITIES = {
    "Tunis": (36.8065, 10.1815),
    "Sfax": (34.7406, 10.7603),
    "Sousse": (35.8256, 10.6369),
    "Monastir": (35.7643, 10.8113),
    "Nabeul": (36.4513, 10.7357),
    "Bizerte": (37.2744, 9.8739),
    "Gabes": (33.8815, 10.0982),
    "Kairouan": (35.6781, 10.0963),
    "Gafsa": (34.4250, 8.7842),
    "Mahdia": (35.5047, 11.0622),
}

# ---------------------------------------------------------------------------
# 2. Referentiel des automates (3 GABs par ville, simules de facon realiste)
# ---------------------------------------------------------------------------
ATMS = {
    # --- Tunis ---
    "ATM_TUNIS_01": (36.8065, 10.1815),   # Centre-ville (Av. Habib Bourguiba)
    "ATM_TUNIS_02": (36.8480, 10.1980),   # Ariana / Ennasr (~5 km au nord)
    "ATM_TUNIS_03": (36.8780, 10.3250),   # La Marsa / Banlieue Nord (~15 km)

    # --- Sfax ---
    "ATM_SFAX_01": (34.7406, 10.7603),    # Centre-ville (Bab Bhar)
    "ATM_SFAX_02": (34.7800, 10.7400),    # Sakiet Ezzit (~5 km)
    "ATM_SFAX_03": (34.6900, 10.7100),    # Thyna / Route de Gabes (~8 km)

    # --- Sousse ---
    "ATM_SOUSSE_01": (35.8256, 10.6369),  # Centre Sousse (Corniche)
    "ATM_SOUSSE_02": (35.8850, 10.5900),  # Port El Kantaoui (~8 km)
    "ATM_SOUSSE_03": (35.7900, 10.6300),  # Sousse Sud / Zaouiet (~4 km)

    # --- Monastir ---
    "ATM_MONASTIR_01": (35.7643, 10.8113), # Centre-ville
    "ATM_MONASTIR_02": (35.7400, 10.7800), # Skanes (~4 km)
    "ATM_MONASTIR_03": (35.7000, 10.8500), # Khniss / Ksibet (~6 km)

    # --- Nabeul ---
    "ATM_NABEUL_01": (36.4513, 10.7357),  # Centre Nabeul
    "ATM_NABEUL_02": (36.4100, 10.6200),  # Hammamet Centre (~12 km)
    "ATM_NABEUL_03": (36.4800, 10.7500),  # Dar Chaabane (~4 km)

    # --- Bizerte ---
    "ATM_BIZERTE_01": (37.2744, 9.8739),  # Vieux Port
    "ATM_BIZERTE_02": (37.2400, 9.8800),  # Zarzouna (~4 km)
    "ATM_BIZERTE_03": (37.3100, 9.8500),  # Corniche Bizerte (~5 km)

    # --- Gabes ---
    "ATM_GABES_01": (33.8815, 10.0982),   # Centre-ville
    "ATM_GABES_02": (33.8500, 10.1200),   # Teboulbou (~4 km)
    "ATM_GABES_03": (33.9200, 10.0500),   # Chenini (~6 km)

    # --- Kairouan ---
    "ATM_KAIROUAN_01": (35.6781, 10.0963), # Medina
    "ATM_KAIROUAN_02": (35.6500, 10.1200), # Zone Sud (~4 km)
    "ATM_KAIROUAN_03": (35.7100, 10.0800), # Zone Nord (~4 km)

    # --- Gafsa ---
    "ATM_GAFSA_01": (34.4250, 8.7842),    # Centre-ville
    "ATM_GAFSA_02": (34.4000, 8.8100),    # El Ksar (~4 km)
    "ATM_GAFSA_03": (34.4600, 8.7600),    # Gafsa Nord (~5 km)

    # --- Mahdia ---
    "ATM_MAHDIA_01": (35.5047, 11.0622),  # Centre-ville
    "ATM_MAHDIA_02": (35.5300, 11.0400),  # Zone Touristique (~4 km)
    "ATM_MAHDIA_03": (35.4700, 11.0500),  # Rejiche (~4 km)
}


# ---------------------------------------------------------------------------
# 3. Resolution des coordonnees (GAB precis puis ville)
# ---------------------------------------------------------------------------
def get_coordinates(atm_id: str = "", city: str = "") -> tuple[float, float] | None:
    """Retourne (latitude, longitude) pour un GAB ou une ville donnee.

    Priorite :
      1. Coordonnees exactes du GAB (atm_id dans ATMS)
      2. Repli sur le centre de la ville (city dans CITIES)
      3. None si ni le GAB ni la ville ne sont trouves
    """
    if atm_id and atm_id in ATMS:
        return ATMS[atm_id]
    if city and city in CITIES:
        return CITIES[city]
    return None


# ---------------------------------------------------------------------------
# 4. Calcul de distance geodesique (Haversine)
# ---------------------------------------------------------------------------
def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance geodesique entre deux points (lat, lon) en kilometres."""
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))
