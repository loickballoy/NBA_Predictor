"""
Accès au bucket, avec deux implémentations interchangeables :
  gs://<bucket>     -> Google Cloud Storage (production, Cloud Run Job)
  file:///chemin    -> un dossier local qui imite le bucket (tests, docker compose)

Le job ne connaît que cette interface : on peut le tester entièrement sans GCP.
"""
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root)

    def list(self, prefix: str) -> list:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())

    def download(self, name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / name, dest)

    def upload(self, src: Path, name: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    def read_json(self, name: str) -> Tuple[Optional[dict], Optional[int]]:
        path = self.root / name
        if not path.exists():
            return None, None
        return json.loads(path.read_text()), None

    def write_json(self, name: str, data: dict, if_generation_match: Optional[int]) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2) + "\n")


class GCSStorage:
    def __init__(self, bucket_name: str):
        from google.cloud import storage   # importé ici : inutile en mode local

        # client.bucket() ne fait AUCUN appel réseau, contrairement à get_bucket(),
        # qui demanderait storage.buckets.get : un droit que trainer n'a pas.
        self.bucket = storage.Client().bucket(bucket_name)

    def list(self, prefix: str) -> list:
        # Le listing avec ce préfixe exact est autorisé par la condition objectListPrefix.
        return sorted(b.name for b in self.bucket.list_blobs(prefix=prefix) if not b.name.endswith("/"))

    def download(self, name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(name).download_to_filename(str(dest))

    def upload(self, src: Path, name: str) -> None:
        self.bucket.blob(name).upload_from_filename(str(src))

    def read_json(self, name: str) -> Tuple[Optional[dict], Optional[int]]:
        blob = self.bucket.get_blob(name)
        if blob is None:
            return None, None
        data = json.loads(blob.download_as_bytes(if_generation_match=blob.generation))
        return data, blob.generation

    def write_json(self, name: str, data: dict, if_generation_match: Optional[int]) -> None:
        # Écriture conditionnelle : si production.json a changé depuis notre lecture
        # (deux jobs en parallèle), GCS refuse l'écriture au lieu d'écraser l'autre.
        # 0 = "seulement si le fichier n'existe pas encore".
        self.bucket.blob(name).upload_from_string(
            json.dumps(data, indent=2) + "\n",
            content_type="application/json",
            if_generation_match=0 if if_generation_match is None else if_generation_match,
        )


def open_storage(uri: str):
    if uri.startswith("gs://"):
        return GCSStorage(uri[len("gs://"):].rstrip("/"))
    if uri.startswith("file://"):
        return LocalStorage(uri[len("file://"):])
    raise ValueError(f"STORAGE_URI non supporté : {uri} (attendu gs://... ou file://...)")
