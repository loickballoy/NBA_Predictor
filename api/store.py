"""
Lecture du modèle en production. L'API ne calcule rien : elle lit ce que le job a produit.

production.json est relu au plus toutes les REFRESH_SECONDS : une promotion par le job
est visible sans redéployer l'API. Le parquet de prédictions n'est rechargé que si la
version change (il est immuable pour une version donnée).
"""
import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline.storage import open_storage

PRODUCTION = "models/production.json"


@dataclass
class Snapshot:
    version: str
    production: dict
    metrics: dict
    predictions: pd.DataFrame          # indexé par id_PLAYER_ID


class ModelStore:
    def __init__(self, storage_uri: str, refresh_seconds: float = 300.0):
        self.storage = open_storage(storage_uri)
        self.refresh_seconds = refresh_seconds
        self._snapshot: Optional[Snapshot] = None
        self._checked_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> Snapshot:
        with self._lock:
            if self._snapshot is None or time.monotonic() - self._checked_at > self.refresh_seconds:
                self._refresh()
            return self._snapshot

    def _refresh(self) -> None:
        production, _ = self.storage.read_json(PRODUCTION)
        if production is None:
            raise LookupError("Aucun modèle en production (models/production.json absent)")
        self._checked_at = time.monotonic()
        if self._snapshot is not None and self._snapshot.version == production["version"]:
            return

        version = production["version"]
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "predictions.parquet"
            self.storage.download(production["predictions"], local)
            predictions = pd.read_parquet(local).set_index("id_PLAYER_ID")
            metrics_path = Path(tmp) / "metrics.json"
            self.storage.download(f"models/{version}/metrics.json", metrics_path)
            metrics = json.loads(metrics_path.read_text())

        self._snapshot = Snapshot(version, production, metrics, predictions)
