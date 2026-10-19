from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str


class StatMetrics(BaseModel):
    mae: Optional[float]
    baseline_mae: Optional[float]
    gain_pct: Optional[float]


class ModelInfo(BaseModel):
    version: str
    promoted_at: str
    validation_score: float = Field(description="Moyenne des MAE modèle / baseline sur la validation (< 1 = meilleur)")
    previous_version: Optional[str]
    architecture: List[int]
    hyperparameters: Dict[str, float]
    test_metrics: Dict[str, StatMetrics] = Field(description="Rapport final sur le test set, jamais utilisé pour décider")


class StatPrediction(BaseModel):
    predicted: float
    last_season: Optional[float] = Field(description="Valeur de la dernière saison (la baseline)")


class PlayerPrediction(BaseModel):
    player_id: int
    player_name: Optional[str]
    season: str = Field(description="Saison prédite, ex. 2026-27")
    model_version: str
    stats: Dict[str, StatPrediction]
