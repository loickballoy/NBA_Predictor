"""
Porte de qualité : un modèle n'est promu que s'il bat la baseline ET le modèle en production.

Score = moyenne, sur les statistiques prédites, de MAE modèle / MAE baseline
("le joueur refait sa dernière saison"). Plus bas = meilleur ; < 1 = meilleur que la baseline.
Le score se calcule toujours sur la VALIDATION : le test set n'intervient jamais ici.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


def score_predictions(pred: pd.DataFrame) -> dict:
    """Calcule le score à partir d'un predictions.csv (colonnes <stat>_true/_pred/_last)."""
    stats = [c[: -len("_true")] for c in pred.columns if c.endswith("_true")]
    if not stats:
        raise ValueError("Aucune colonne *_true : impossible d'évaluer")

    per_stat = {}
    ratios = []
    for s in stats:
        mae = (pred[f"{s}_pred"] - pred[f"{s}_true"]).abs().mean()
        entry = {"mae": float(mae)}
        if f"{s}_last" in pred.columns:
            base = (pred[f"{s}_last"] - pred[f"{s}_true"]).abs().mean()
            entry["baseline_mae"] = float(base)
            if base > 0:
                ratios.append(mae / base)
        per_stat[s] = entry

    if not ratios:
        raise ValueError("Aucune baseline disponible : impossible de calculer le score")
    return {"score": float(sum(ratios) / len(ratios)), "n_examples": int(len(pred)), "per_stat": per_stat}


@dataclass
class Decision:
    promote: bool
    reason: str


def decide(challenger: float, champion: Optional[float], min_improvement: float = 0.0) -> Decision:
    """challenger : score du nouveau modèle. champion : score du modèle en production
    sur la MÊME validation, ou None s'il n'y en a pas (ou s'il est incompatible).
    min_improvement : gain relatif minimum exigé face au champion (0.01 = 1 %)."""
    if challenger >= 1.0:
        return Decision(False, f"ne bat pas la baseline (score {challenger:.4f} >= 1)")
    if champion is None:
        return Decision(True, f"bat la baseline (score {challenger:.4f}) et aucun modèle en production")
    threshold = champion * (1.0 - min_improvement)
    if challenger < threshold:
        return Decision(True, f"bat la baseline et le modèle en production ({challenger:.4f} < {threshold:.4f})")
    return Decision(False, f"ne bat pas le modèle en production ({challenger:.4f} >= {threshold:.4f})")
