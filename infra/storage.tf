resource "google_storage_bucket" "mlops" {
  name     = var.bucket_name
  location = upper(var.region)

  uniform_bucket_level_access = true      # droits uniquement par IAM, nécessaire aux conditions IAM
  public_access_prevention    = "enforced" # le bucket ne peut jamais devenir public

  # Projet de démonstration recréable : destroy doit pouvoir tout supprimer.
  # Les données se recopient depuis le homelab.
  force_destroy = var.allow_destroy_data

  versioning {
    enabled = true
  }

  # Versions remplacées ou supprimées : conservées 30 jours, puis effacées.
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }

  # Le versioning protège déjà les données : pas besoin de payer en plus
  # les 7 jours de rétention "soft delete" activés par défaut par GCS.
  soft_delete_policy {
    retention_duration_seconds = 0
  }

  labels = {
    component = "storage"
  }
}
