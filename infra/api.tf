# API de prédiction. Pour l'instant : authentification IAM obligatoire (aucun accès public).
# IAP sera activé au bloc 3 de la phase 4.
resource "google_cloud_run_v2_service" "api" {
  name                = "nba-api"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  iap_enabled         = true

  labels = {
    component = "api"
  }

  template {
    service_account = google_service_account.api.email

    # FinOps : 0 instance quand personne n'appelle l'API, 2 au maximum.
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image   = var.api_image != "" ? var.api_image : var.trainer_image
      command = ["uvicorn"]
      args    = ["api.main:app", "--host", "0.0.0.0", "--port", "8080"]

      ports {
        container_port = 8080
      }

      env {
        name  = "STORAGE_URI"
        value = "gs://${google_storage_bucket.mlops.name}"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true # CPU facturé uniquement pendant le traitement des requêtes
      }

      startup_probe {
        http_get {
          path = "/health"
        }
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

data "google_project" "this" {}

# IAP doit avoir le droit d'appeler le service une fois l'utilisateur authentifié.
resource "google_cloud_run_v2_service_iam_member" "iap_invoker" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-iap.iam.gserviceaccount.com"
}

# Les seules personnes autorisées à passer IAP.
resource "google_iap_web_cloud_run_service_iam_member" "users" {
  for_each               = toset(var.iap_users)
  location               = google_cloud_run_v2_service.api.location
  cloud_run_service_name = google_cloud_run_v2_service.api.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value
}