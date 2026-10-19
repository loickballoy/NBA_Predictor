locals {
  objects = "projects/_/buckets/${google_storage_bucket.mlops.name}/objects/"

  # Condition "lire et lister un préfixe".
  # Lister un dossier est une opération sur le BUCKET, pas sur un objet : resource.name
  # vaut alors le nom du bucket et le startsWith() échoue. L'attribut objectListPrefix
  # contient le préfixe demandé dans la requête de listing, ce qui permet de l'autoriser.
  read_and_list = {
    for p in ["data", "models"] :
    p => "resource.name.startsWith(\"${local.objects}${p}/\") || api.getAttribute(\"storage.googleapis.com/objectListPrefix\", \"\").startsWith(\"${p}/\")"
  }
}

# ---------- trainer ----------

# Lit les données d'entraînement (et liste les fichiers parquet de data/).
resource "google_storage_bucket_iam_member" "trainer_read_data" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.trainer.email}"

  condition {
    title      = "trainer-data-read"
    expression = local.read_and_list["data"]
  }
}

# Lit, écrit et ÉCRASE dans models/ : la porte de qualité lit les métriques du modèle
# en production et met à jour models/production.json. objectCreator ne suffit pas,
# il interdit de remplacer un fichier existant. objectUser autorise aussi la suppression :
# le versioning du bucket rend toute suppression récupérable pendant 30 jours.
resource "google_storage_bucket_iam_member" "trainer_rw_models" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.trainer.email}"

  condition {
    title      = "trainer-models-rw"
    expression = local.read_and_list["models"]
  }
}

# Crée les fichiers de prédictions. Chaque fichier porte la version du modèle dans son
# nom (predictions/<version>.parquet) : jamais besoin d'écraser, objectCreator suffit.
resource "google_storage_bucket_iam_member" "trainer_write_predictions" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.trainer.email}"

  condition {
    title      = "trainer-predictions-create"
    expression = "resource.name.startsWith(\"${local.objects}predictions/\")"
  }
}

# ---------- api ----------
# Lecture seule, par nom exact (production.json, puis predictions/<version>.parquet) :
# aucun besoin de lister, donc pas d'objectListPrefix.

resource "google_storage_bucket_iam_member" "api_read_models" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"

  condition {
    title      = "api-models-read"
    expression = "resource.name.startsWith(\"${local.objects}models/\")"
  }
}

resource "google_storage_bucket_iam_member" "api_read_predictions" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"

  condition {
    title      = "api-predictions-read"
    expression = "resource.name.startsWith(\"${local.objects}predictions/\")"
  }
}

# Accès à ce secret uniquement, pas à tous les secrets du projet.
resource "google_secret_manager_secret_iam_member" "api_read_anthropic" {
  secret_id = google_secret_manager_secret.anthropic_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# ---------- deployer ----------
# Pousse les images. writer inclut reader, nécessaire aussi pour déployer une image.
# Les droits de déploiement Cloud Run seront ajoutés en phase 5.
#
# Pas de droit Artifact Registry pour trainer et api : dans le même projet, c'est l'agent
# de service Cloud Run (service-<n°>@serverless-robot-prod) qui télécharge l'image,
# pas le compte d'exécution. Leur donner reader serait un privilège inutile.
resource "google_artifact_registry_repository_iam_member" "deployer_push" {
  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}
