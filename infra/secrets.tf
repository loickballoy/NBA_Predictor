# Seul le "contenant" du secret est créé ici. La clé elle-même est ajoutée à la main :
#   printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets versions add anthropic-api-key --data-file=-
# Ainsi elle n'apparaît ni dans le dépôt, ni dans l'état Terraform.
resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = var.anthropic_secret_id

  replication {
    auto {}
  }

  labels = {
    component = "secrets"
  }

  depends_on = [google_project_service.apis]
}
