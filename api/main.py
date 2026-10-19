"""
API de prédiction : `uvicorn api.main:app --host 0.0.0.0 --port 8080`

  GET /health                      sonde de disponibilité (ne touche pas au bucket)
  GET /model                       modèle en production : version, date, métriques
  GET /players/{id}/prediction     prédiction de la saison prochaine pour un joueur
"""
import json
import math
import os
import time

from fastapi import FastAPI, HTTPException, Request
from nba_api.stats.static import players as nba_players

from api.schemas import Health, ModelInfo, PlayerPrediction, StatMetrics, StatPrediction
from api.store import ModelStore

store = ModelStore(os.environ.get("STORAGE_URI", "file:///tmp/nba-bucket"),
                   refresh_seconds=float(os.environ.get("REFRESH_SECONDS", 300)))

app = FastAPI(title="NBA Predictor", version="1.0")


def log(severity: str, message: str, **fields) -> None:
    print(json.dumps({"severity": severity, "message": message, "component": "api", **fields}, ensure_ascii=False), flush=True)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Une ligne de log structurée par requête : endpoint, statut, latence, version du modèle."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        log("INFO" if status < 500 else "ERROR", "requête",
            event="request", method=request.method,
            endpoint=getattr(route, "path", request.url.path),   # /players/{player_id}/prediction
            status=status, latency_ms=round((time.perf_counter() - start) * 1000, 1),
            model_version=getattr(request.state, "model_version", None))


def snapshot(request: Request):
    try:
        snap = store.get()
    except LookupError as e:
        raise HTTPException(status_code=503, detail=str(e))
    request.state.model_version = snap.version
    return snap


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok")


@app.get("/model", response_model=ModelInfo)
def model(request: Request):
    snap = snapshot(request)
    test = snap.metrics["test"]
    stats = test["mae"].keys()
    return ModelInfo(
        version=snap.version,
        promoted_at=snap.production["promoted_at"],
        validation_score=snap.production["validation_score"],
        previous_version=snap.production.get("previous_version"),
        architecture=snap.metrics["model"]["architecture"],
        hyperparameters=snap.metrics["hyperparameters"],
        test_metrics={s: StatMetrics(mae=clean(test["mae"][s]), baseline_mae=clean(test["baseline_mae"][s]),
                                     gain_pct=clean(test["gain_pct"][s])) for s in stats},
    )


@app.get("/players/{player_id}/prediction", response_model=PlayerPrediction)
def prediction(player_id: int, request: Request):
    snap = snapshot(request)
    if player_id not in snap.predictions.index:
        raise HTTPException(status_code=404,
                            detail="Joueur sans prédiction (il faut 3 saisons consécutives jusqu'à la dernière)")
    row = snap.predictions.loc[player_id]
    year = int(row["id_TARGET_YEAR"])
    stats = [c[: -len("_pred")] for c in snap.predictions.columns if c.endswith("_pred")]
    info = nba_players.find_player_by_id(player_id)   # base locale de nba_api : aucun appel réseau

    return PlayerPrediction(
        player_id=player_id,
        player_name=info["full_name"] if info else None,
        season=f"{year}-{str(year + 1)[2:]}",
        model_version=snap.version,
        stats={s: StatPrediction(predicted=round(float(row[f"{s}_pred"]), 2),
                                 last_season=clean(row.get(f"{s}_last"))) for s in stats},
    )
