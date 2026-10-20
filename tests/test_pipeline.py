"""Tests de la porte de qualité, des fenêtres à prédire et du job complet (stockage local)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from collector.build_dataset import build_inference_windows
from pipeline.gate import decide, score_predictions
from tests.test_collector import seasons

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "model" / "bin"


# ---------- porte de qualité ----------

def test_refuse_if_not_better_than_baseline():
    assert not decide(1.0, None).promote
    assert not decide(1.2, 1.5).promote      # même meilleur que la prod, la baseline prime


def test_promote_first_model_if_better_than_baseline():
    assert decide(0.8, None).promote


def test_promote_only_if_strictly_better_than_production():
    assert decide(0.79, 0.80).promote
    assert not decide(0.80, 0.80).promote    # égalité : on garde le modèle en place
    assert not decide(0.81, 0.80).promote


def test_min_improvement_is_required():
    assert not decide(0.795, 0.80, min_improvement=0.01).promote   # -0.6 % < 1 % exigé
    assert decide(0.78, 0.80, min_improvement=0.01).promote


def test_score_is_mean_ratio_to_baseline():
    pred = pd.DataFrame({
        "PTS_true": [10.0, 20.0], "PTS_pred": [11.0, 19.0], "PTS_last": [12.0, 18.0],  # 1 / 2
        "GP_true": [70.0, 60.0], "GP_pred": [70.0, 60.0], "GP_last": [60.0, 70.0],     # 0 / 10
    })
    assert score_predictions(pred)["score"] == pytest.approx((0.5 + 0.0) / 2)


# ---------- fenêtres à prédire ----------

def test_inference_windows_predict_the_season_after_the_last_one():
    df = seasons({1: range(2018, 2026), 2: range(2015, 2022)})   # le joueur 2 a arrêté en 2021-22
    w = build_inference_windows(df)
    assert w["id_PLAYER_ID"].tolist() == [1]                    # seuls les joueurs actifs
    assert w["id_TARGET_YEAR"].tolist() == [2026]
    assert w["x_s3_GP"].iloc[0] + 1990 == 2025                  # dernière saison connue en entrée
    assert not any(c.startswith("y_") for c in w.columns)


# ---------- job complet, bucket simulé par un dossier ----------

def fake_raw_data(n_players=300, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    career, awards = [], []
    for pid in range(1, n_players + 1):
        start = int(rng.integers(1998, 2020))
        talent = rng.gamma(2, 4)
        for s in range(int(rng.integers(4, 14))):
            y = start + s
            if y > 2025:
                break
            gp = int(np.clip(rng.normal(65, 12), 5, 82))
            ppg = max(0.0, talent * (1 + 0.08 * s - 0.01 * s * s) + rng.normal(0, 1.5))
            career.append(dict(
                PLAYER_ID=pid, SEASON_ID=f"{y}-{str(y + 1)[2:]}", LEAGUE_ID="00", TEAM_ID=1,
                TEAM_ABBREVIATION="BOS", PLAYER_AGE=float(20 + s), GP=gp, GS=gp // 2, MIN=gp * (8 + ppg),
                FGM=gp * ppg * .38, FGA=gp * ppg * .85, FG_PCT=.45, FG3M=gp * ppg * .1, FG3A=gp * ppg * .28,
                FG3_PCT=.35, FTM=gp * ppg * .15, FTA=gp * ppg * .2, FT_PCT=.77, OREB=gp * .5,
                DREB=gp * ppg * .25, REB=gp * ppg * .3, AST=gp * ppg * .2, STL=gp * .6, BLK=gp * .3,
                TOV=gp * ppg * .1, PF=gp * 1.8, PTS=gp * ppg))
            if ppg > 22:
                awards.append(dict(PERSON_ID=pid, FIRST_NAME="A", LAST_NAME="B", DESCRIPTION="NBA All-Star",
                                   ALL_NBA_TEAM_NUMBER=None, SEASON=f"{y}-{str(y + 1)[2:]}"))
    return pd.DataFrame(career), pd.DataFrame(awards)


@pytest.mark.skipif(not (BIN / "train").exists(), reason="binaires C++ non compilés (make -C model)")
def test_job_promotes_then_refuses_an_identical_model(tmp_path):
    bucket = tmp_path / "bucket"
    career, awards = fake_raw_data()
    (bucket / "data/raw/career_stats").mkdir(parents=True)
    (bucket / "data/raw/awards").mkdir(parents=True)
    career.to_parquet(bucket / "data/raw/career_stats/all.parquet")
    awards.to_parquet(bucket / "data/raw/awards/all.parquet")

    env = {**os.environ, "STORAGE_URI": f"file://{bucket}", "GIT_SHA": "abc1234", "EPOCHS": "40",
           "SEED": "1", "TRAIN_BIN": str(BIN / "train"), "PREDICT_BIN": str(BIN / "predict")}

    def run_job():
        r = subprocess.run([sys.executable, "-m", "pipeline.job"], cwd=REPO, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:]
        events = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        return next(e for e in events if e.get("event") == "gate_decision")

    first = run_job()
    assert first["promoted"], first["reason"]
    prod = json.loads((bucket / "models/production.json").read_text())
    assert prod["version"] == first["version"]
    preds = pd.read_parquet(bucket / prod["predictions"])
    assert (preds["id_TARGET_YEAR"] == 2026).all() and "PTS_PER_GAME_pred" in preds.columns

    # Même données, même graine : même modèle, donc pas strictement meilleur -> refusé.
    import time; time.sleep(1.1)   # versions horodatées à la seconde
    second = run_job()
    assert not second["promoted"]
    assert json.loads((bucket / "models/production.json").read_text())["version"] == first["version"]
    # Le candidat refusé est quand même archivé, avec la raison du refus.
    decision = json.loads((bucket / f"models/{second['version']}/decision.json").read_text())
    assert decision["promoted"] is False and "production" in decision["reason"]
    assert not (bucket / f"predictions/{second['version']}.parquet").exists()
