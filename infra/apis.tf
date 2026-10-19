# Sur un projet neuf, ces API sont désactivées : sans elles, chaque ressource échoue.
# disable_on_destroy = false : destroy supprime les ressources mais laisse les API actives
# (les désactiver est lent, ne coûte rien de plus si on les laisse, et peut casser un
# second apply lancé juste après).
locals {
  apis = [
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "cloudscheduler.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)

  service            = each.value
  disable_on_destroy = false
}
