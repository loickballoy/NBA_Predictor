resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.artifact_registry_name
  description   = "Images Docker du projet NBA MLOps"
  format        = "DOCKER"

  # FinOps : chaque push ajoute une image facturée au stockage.
  # On garde les 5 plus récentes et on supprime le reste après 30 jours.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "30d"
    }
  }

  labels = {
    component = "artifact-registry"
  }

  depends_on = [google_project_service.apis]
}
