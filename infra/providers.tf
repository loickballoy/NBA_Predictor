terraform {
  required_version = ">= 1.9.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # L'état vit dans le bucket créé à la main avant le premier `terraform init`
  # (Terraform ne peut pas créer le bucket qui stocke son propre état).
  # Un backend ne peut pas utiliser de variables : le nom est écrit en dur.
  backend "gcs" {
    bucket = "nba-mlops-lb26-tfstate"
    prefix = "infra"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  # Appliqués automatiquement à toutes les ressources qui acceptent des labels.
  # Chaque ressource n'ajoute que son label `component`.
  default_labels = local.common_labels
}
