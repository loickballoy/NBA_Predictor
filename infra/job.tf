# ---------- Job d'entraînement ----------
resource "google_cloud_run_v2_job" "trainer" {
  name                = "nba-trainer"
  location            = var.region
  deletion_protection = false # sinon terraform destroy refuse de supprimer le job

  labels = {
    component = "trainer"
  }

  template {
    task_count = 1

    template {
      service_account = google_service_account.trainer.email
      timeout         = "3600s"
      max_retries     = 0 # un entraînement raté se relance à la main, après lecture des logs

      containers {
        image   = var.trainer_image
        command = ["python", "-m", "pipeline.job"]

        env {
          name  = "STORAGE_URI"
          value = "gs://${google_storage_bucket.mlops.name}"
        }
        env {
          name  = "LAST_COMPLETE_SEASON"
          value = tostring(var.last_complete_season)
        }

        resources {
          limits = {
            cpu    = "1" # le C++ est mono-thread : un 2e CPU serait payé pour rien
            memory = "2Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------- Planification ----------
# Compte dédié : Cloud Scheduler n'a besoin que de lancer CE job, rien d'autre.
resource "google_service_account" "scheduler" {
  account_id   = "nba-scheduler"
  display_name = "NBA MLOps Scheduler"
  description  = "component=scheduler : lance le job d'entraînement, aucun accès aux données"

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_run_trainer" {
  location = google_cloud_run_v2_job.trainer.location
  name     = google_cloud_run_v2_job.trainer.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "weekly_training" {
  name             = "nba-weekly-training"
  region           = var.scheduler_region
  description      = "Entraînement hebdomadaire pendant la saison NBA"
  schedule         = var.training_schedule
  time_zone        = "Europe/Paris"
  attempt_deadline = "320s"
  paused           = var.schedule_paused

  # Appel de l'API Cloud Run "exécuter le job", authentifié avec le compte scheduler.
  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.trainer.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler_run_trainer]
}
