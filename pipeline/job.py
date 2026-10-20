"""
Job d'entraînement (Cloud Run Job) : `python -m pipeline.job`

  1. télécharge data/raw/ depuis le bucket
  2. construit le dataset (train/val/test + fenêtres à prédire)
  3. entraîne un modèle candidat (binaire C++ `train`)
  4. évalue le candidat ET le modèle en production sur la même validation
  5. porte de qualité : promotion seulement s'il bat la baseline et la production
  6. écrit toujours models/<version>/ (même refusé : traçabilité)
  7. si promu : predictions/<version>.parquet, PUIS models/production.json

Configuration par variables d'environnement (voir Config). Logs en JSON sur stdout :
Cloud Logging les lit comme des entrées structurées (jsonPayload + severity).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from collector.build_dataset import build_inference_windows, build_windows, export_inference, export_splits
from collector.clean import make_season_table
from pipeline.gate import decide, score_predictions
from pipeline.storage import open_storage

PRODUCTION = "models/production.json"


def log(severity: str, message: str, **fields) -> None:
    print(json.dumps({"severity": severity, "message": message, "component": "trainer", **fields}, ensure_ascii=False), flush=True)


@dataclass
class Config:
    storage_uri: str
    git_sha: str = "unknown"
    seed: int = 42
    epochs: int = 1000
    learning_rate: float = 0.001
    val_from: int = 2020
    test_from: int = 2023
    min_improvement: float = 0.0
    last_complete_season: Optional[int] = None   # ex. 2025 : ignore la saison en cours
    train_bin: str = "train"
    predict_bin: str = "predict"

    @staticmethod
    def from_env() -> "Config":
        env = os.environ
        if "STORAGE_URI" not in env:
            raise SystemExit("STORAGE_URI manquant (gs://<bucket> ou file:///chemin)")
        last = env.get("LAST_COMPLETE_SEASON", "")
        return Config(
            storage_uri=env["STORAGE_URI"],
            git_sha=env.get("GIT_SHA", "unknown"),
            seed=int(env.get("SEED", 42)),
            epochs=int(env.get("EPOCHS", 1000)),
            learning_rate=float(env.get("LEARNING_RATE", 0.001)),
            val_from=int(env.get("VAL_FROM", 2020)),
            test_from=int(env.get("TEST_FROM", 2023)),
            min_improvement=float(env.get("MIN_IMPROVEMENT", 0.0)),
            last_complete_season=int(last) if last else None,
            train_bin=env.get("TRAIN_BIN", "train"),
            predict_bin=env.get("PREDICT_BIN", "predict"),
        )


def run(cmd: list) -> None:
    """Lance un binaire C++ ; sa sortie part dans les logs, un code != 0 arrête le job."""
    log("INFO", "commande", cmd=" ".join(map(str, cmd)))
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        print(line, flush=True)
    if result.returncode != 0:
        log("ERROR", "commande échouée", cmd=str(cmd[0]), stderr=result.stderr[-2000:])
        raise RuntimeError(f"{cmd[0]} a échoué (code {result.returncode})")


def load_raw(storage, work: Path) -> tuple:
    raw = work / "raw"
    names = storage.list("data/raw/")
    if not names:
        raise RuntimeError("Aucun fichier sous data/raw/ dans le bucket")
    for name in names:
        storage.download(name, raw / name[len("data/raw/"):])
    log("INFO", "données téléchargées", files=len(names))

    def concat(kind: str) -> pd.DataFrame:
        files = sorted((raw / kind).glob("*.parquet"))
        if not files:
            raise RuntimeError(f"Aucun parquet dans data/raw/{kind}/")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    return concat("career_stats"), concat("awards")


def evaluate(cfg: Config, model_dir: Path, val_csv: Path, out_csv: Path) -> dict:
    run([cfg.predict_bin, "--model", model_dir, "--input", val_csv, "--output", out_csv])
    return score_predictions(pd.read_csv(out_csv))


def champion_score(cfg: Config, storage, work: Path, val_csv: Path) -> tuple:
    """Réévalue le modèle en production sur la validation ACTUELLE.
    Comparer des métriques calculées sur deux validations différentes serait faux."""
    prod, generation = storage.read_json(PRODUCTION)
    if prod is None:
        return None, None, None
    version = prod["version"]
    champ = work / "champion"
    for f in ["model.bin", "scalers.json"]:
        storage.download(f"models/{version}/{f}", champ / f)
    try:
        result = evaluate(cfg, champ, val_csv, work / "champion_val.csv")
    except RuntimeError:
        # Par exemple : les features ont changé, l'ancien modèle ne peut plus lire les données.
        log("WARNING", "modèle en production incompatible avec les données actuelles", version=version)
        return version, None, generation
    return version, result["score"], generation


def main() -> int:
    cfg = Config.from_env()
    version = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{cfg.git_sha[:7]}"
    started = time.time()
    log("INFO", "job démarré", version=version, config=asdict(cfg))

    try:
        storage = open_storage(cfg.storage_uri)
        work = Path(tempfile.mkdtemp(prefix="nba-job-"))
        processed = work / "processed"
        candidate = work / "models" / version

        # 1-2. Données et dataset
        career, awards = load_raw(storage, work)
        seasons = make_season_table(career, awards)
        if cfg.last_complete_season is not None:
            start_year = seasons["SEASON_ID"].str[:4].astype(int)
            seasons = seasons[start_year <= cfg.last_complete_season]
        export_splits(build_windows(seasons), cfg.val_from, cfg.test_from, out_dir=processed)
        export_inference(build_inference_windows(seasons), out_dir=processed)
        val_csv = processed / "val.csv"

        # 3. Entraînement du candidat
        run([cfg.train_bin, "--data", processed, "--out", candidate, "--seed", cfg.seed,
             "--epochs", cfg.epochs, "--lr", cfg.learning_rate])

        # 4. Évaluation du candidat et du champion sur la même validation
        challenger = evaluate(cfg, candidate, val_csv, work / "challenger_val.csv")
        champ_version, champ, generation = champion_score(cfg, storage, work, val_csv)

        # 5. Porte de qualité
        decision = decide(challenger["score"], champ, cfg.min_improvement)
        record = {
            "version": version,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "promoted": decision.promote,
            "reason": decision.reason,
            "challenger": challenger,
            "champion": {"version": champ_version, "score": champ},
            "min_improvement": cfg.min_improvement,
        }
        (candidate / "decision.json").write_text(json.dumps(record, indent=2) + "\n")

        # 6. Toujours archiver le candidat, promu ou non
        for f in sorted(candidate.iterdir()):
            storage.upload(f, f"models/{version}/{f.name}")

        # 7. Promotion : les prédictions d'abord, le pointeur ensuite.
        #    Ainsi production.json ne désigne jamais une version sans prédictions.
        if decision.promote:
            pred_csv = work / "inference_predictions.csv"
            run([cfg.predict_bin, "--model", candidate, "--input", processed / "inference.csv",
                 "--output", pred_csv])
            # Prédictions + les 3 saisons d'entrée : l'API (et /explain) a tout dans un seul fichier.
            pred = pd.read_csv(pred_csv)
            inputs = pd.read_csv(processed / "inference.csv")
            ids = [c for c in pred.columns if c.startswith("id_")]
            pred = pred.merge(inputs, on=ids, how="left", validate="one_to_one")
            pred_parquet = work / f"{version}.parquet"
            pred.to_parquet(pred_parquet, index=False)
            storage.upload(pred_parquet, f"predictions/{version}.parquet")

            storage.write_json(PRODUCTION, {
                "version": version,
                "promoted_at": record["decided_at"],
                "validation_score": challenger["score"],
                "predictions": f"predictions/{version}.parquet",
                "previous_version": champ_version,
            }, if_generation_match=generation)

        log("NOTICE" if decision.promote else "WARNING",
            "modèle promu" if decision.promote else "modèle refusé",
            event="gate_decision", version=version, promoted=decision.promote, reason=decision.reason,
            challenger_score=challenger["score"], champion_score=champ, champion_version=champ_version,
            duration_s=round(time.time() - started, 1))
        return 0

    except Exception as e:   # noqa: BLE001 - tout échec doit être journalisé puis faire échouer le job
        log("ERROR", "job échoué", event="job_failed", version=version, error=f"{type(e).__name__}: {e}",
            duration_s=round(time.time() - started, 1))
        return 1


if __name__ == "__main__":
    sys.exit(main())
