variable "project_id" {
  type        = string
  description = "ID du projet GCP"
}

variable "region" {
  type        = string
  default     = "europe-west9"
  description = "Région GCP (europe-west9 = Paris)"
}

variable "bucket_name" {
  type        = string
  description = "Nom du bucket de données et de modèles (unique au niveau mondial)"
}

variable "artifact_registry_name" {
  type        = string
  default     = "nba-mlops"
  description = "Nom du dépôt Docker dans Artifact Registry"
}

variable "subnet_cidr" {
  type        = string
  default     = "10.10.0.0/24"
  description = "Plage d'adresses du sous-réseau"
}

variable "anthropic_secret_id" {
  type        = string
  default     = "anthropic-api-key"
  description = "Nom du secret contenant la clé API Anthropic"
}

variable "allow_destroy_data" {
  type        = bool
  default     = true
  description = "Autorise terraform destroy à supprimer le bucket même s'il contient des fichiers"
}

variable "trainer_image" {
  type        = string
  description = "Image complète du job, ex. europe-west9-docker.pkg.dev/<projet>/nba-mlops/nba-predictor:<version>"
}

variable "last_complete_season" {
  type        = number
  default     = 2025
  description = "Dernière saison terminée (année de début) : la saison en cours est ignorée"
}

variable "training_schedule" {
  type        = string
  default     = "0 6 * * 1"
  description = "Fréquence d'entraînement (cron) : le lundi à 6 h"
}

variable "scheduler_region" {
  type        = string
  default     = "europe-west1"
  description = "Région de Cloud Scheduler (n'a pas besoin d'être celle du job)"
}

variable "schedule_paused" {
  type        = bool
  default     = true
  description = "Planification en pause par défaut : on l'active pendant la saison NBA"
}

variable "api_image" {
  type        = string
  default     = ""
  description = "Image de l'API ; vide = même image que le job"
}

variable "iap_users" {
  type        = list(string)
  description = "Comptes autorisés à ouvrir l'API, ex. [\"user:moi@gmail.com\"]"
}

locals {
  common_labels = {
    project    = "nba-mlops"
    managed_by = "terraform"
  }
}
