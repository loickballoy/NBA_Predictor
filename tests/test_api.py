"""Tests de l'API, sur un bucket simulé rempli par une vraie exécution du job."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
from api.store import ModelStore
from tests.test_pipeline import BIN, REPO, fake_raw_data


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    if not (BIN / "train").exists():
        pytest.skip("binaires C++ non compilés (make -C model)")
    root = tmp_path_factory.mktemp("bucket")
    career, awards = fake_raw_data()
    for kind, df in [("career_stats", career), ("awards", awards)]:
        (root / f"data/raw/{kind}").mkdir(parents=True)
        df.to_parquet(root / f"data/raw/{kind}/all.parquet")
    env = {**os.environ, "STORAGE_URI": f"file://{root}", "GIT_SHA": "abc1234", "EPOCHS": "30",
           "TRAIN_BIN": str(BIN / "train"), "PREDICT_BIN": str(BIN / "predict")}
    r = subprocess.run([sys.executable, "-m", "pipeline.job"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    return root


@pytest.fixture
def client(bucket, monkeypatch):
    monkeypatch.setattr(api.main, "store", ModelStore(f"file://{bucket}", refresh_seconds=0))
    return TestClient(api.main.app)


def production(bucket):
    return json.loads((bucket / "models/production.json").read_text())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_describes_production(client, bucket):
    body = client.get("/model").json()
    assert body["version"] == production(bucket)["version"]
    assert body["validation_score"] < 1
    assert set(body["test_metrics"]) >= {"GP", "PTS_PER_GAME"}


def test_prediction_for_known_player(client, bucket):
    preds = pd.read_parquet(bucket / production(bucket)["predictions"])
    player_id = int(preds["id_PLAYER_ID"].iloc[0])

    body = client.get(f"/players/{player_id}/prediction").json()
    assert body["season"] == "2026-27"
    assert body["model_version"] == production(bucket)["version"]
    assert len(body["stats"]) == 8
    assert body["stats"]["PTS_PER_GAME"]["last_season"] is not None


def test_unknown_player_is_404(client):
    assert client.get("/players/999999999/prediction").status_code == 404


def test_no_production_model_is_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api.main, "store", ModelStore(f"file://{tmp_path}"))
    assert TestClient(api.main.app).get("/model").status_code == 503


def test_promotion_is_picked_up_without_restart(client, bucket):
    first = client.get("/model").json()["version"]
    prod = production(bucket)
    # Simule une nouvelle promotion : même fichiers, nouveau nom de version.
    new = "20991231-000000-newmodel"
    (bucket / f"models/{new}").mkdir()
    for f in (bucket / f"models/{first}").iterdir():
        (bucket / f"models/{new}/{f.name}").write_bytes(f.read_bytes())
    (bucket / f"predictions/{new}.parquet").write_bytes((bucket / prod["predictions"]).read_bytes())
    (bucket / "models/production.json").write_text(json.dumps(
        {**prod, "version": new, "predictions": f"predictions/{new}.parquet"}))

    assert client.get("/model").json()["version"] == new   # refresh_seconds=0 : relu à chaque appel
