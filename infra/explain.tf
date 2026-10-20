# Cache des explications de Claude : l'API lit et CRÉE sous explanations/<version>/<joueur>.json.
# Jamais d'écrasement (objectCreator) : une explication validée pour une version ne change plus.
resource "google_storage_bucket_iam_member" "api_read_explanations" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"

  condition {
    title      = "api-explanations-read"
    expression = "resource.name.startsWith(\"${local.objects}explanations/\")"
  }
}

resource "google_storage_bucket_iam_member" "api_create_explanations" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.api.email}"

  condition {
    title      = "api-explanations-create"
    expression = "resource.name.startsWith(\"${local.objects}explanations/\")"
  }
}
