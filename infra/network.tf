# Les réseaux et sous-réseaux n'acceptent pas de labels dans GCP. Ils sont gratuits :
# ils n'apparaîtront donc jamais dans la ventilation des coûts par composant.
resource "google_compute_network" "vpc" {
  name                    = "nba-mlops-vpc"
  auto_create_subnetworks = false
  description             = "VPC du projet NBA MLOps (component=network)"

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "main" {
  name                     = "nba-mlops-subnet"
  region                   = var.region
  network                  = google_compute_network.vpc.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
  description              = "Sous-réseau principal NBA MLOps (component=network)"
}
