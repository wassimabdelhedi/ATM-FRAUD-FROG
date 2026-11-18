"""Moteur de decision commun partage entre Fog A, Fog B et Fog C.

Ce module est IDENTIQUE sur les 3 noeuds : garantit que A et B
prennent exactement la meme decision que C aurait prise, en cas
de panne de C. Pas de contradiction possible entre noeuds.

Seuils calibres sur les resultats de validation du modele Random Forest :
  - behavior_score >= 0.03  : seuil optimal (minimize le cout financier)
  - behavior_score >= 0.015 : zone grise (moitie du seuil optimal)
"""


def make_decision(behavior_score: float, impossible_travel: bool) -> dict:
    """Applique la politique de decision unifiee.

    Args:
        behavior_score: Probabilite de fraude calculee par le Random Forest [0..1]
        impossible_travel: True si le deplacement inter-villes est physiquement impossible

    Returns:
        dict avec les cles : decision (BLOCK/REVIEW/ALLOW), reason, details
    """

    # ----------------------------------------------------------------
    # Regle 1 : Preuve physique absolue (Clonage / Voyage impossible)
    # C'est une preuve deterministe, pas probabiliste.
    # Un humain ne peut pas parcourir 300 km en 5 minutes.
    # ----------------------------------------------------------------
    if impossible_travel:
        return {
            "decision": "BLOCK",
            "reason": "FRAUD_IMPOSSIBLE_TRAVEL",
            "details": "Deplacement physiquement impossible entre deux villes detecte (clonage de carte).",
            "action": "Carte bloquee immediatement. Alerte transmise a la banque.",
        }

    # ----------------------------------------------------------------
    # Regle 2 : Anomalie comportementale critique (IA Random Forest)
    # Score >= seuil optimal calibre sur validation (0.03).
    # Le modele a detecte une rafale ou un montant anormalement eleve.
    # ----------------------------------------------------------------
    if behavior_score >= 0.03:
        return {
            "decision": "BLOCK",
            "reason": "FRAUD_BEHAVIOR_ANOMALY",
            "details": f"Score comportemental IA = {behavior_score:.4f} (seuil 0.03). Rafale ou montant atypique detecte.",
            "action": "Retrait refuse. Carte temporairement suspendue en attente de verification.",
        }

    # ----------------------------------------------------------------
    # Regle 3 : Zone grise / Suspicion legere (Demi-seuil)
    # Score entre 0.015 et 0.03 : comportement inhabituel mais
    # pas assez fort pour bloquer definitivement.
    # Action : demander une confirmation supplementaire au porteur.
    # ----------------------------------------------------------------
    if behavior_score >= 0.015:
        return {
            "decision": "REVIEW",
            "reason": "SUSPICIOUS_BEHAVIOR",
            "details": f"Score comportemental IA = {behavior_score:.4f} (zone grise 0.015-0.03). Comportement inhabituel.",
            "action": "Demande de confirmation par code OTP SMS au porteur avant autorisation.",
        }

    # ----------------------------------------------------------------
    # Regle 4 : Transaction saine et coherente
    # ----------------------------------------------------------------
    return {
        "decision": "ALLOW",
        "reason": "LEGITIMATE",
        "details": f"Score comportemental IA = {behavior_score:.4f} (< 0.015). Deplacement coherent.",
        "action": "Distribution des billets autorisee.",
    }
