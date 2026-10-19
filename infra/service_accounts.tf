# Les comptes de service n'acceptent pas de labels dans GCP, et ils sont gratuits.
# Le composant figure dans la description pour rester lisible dans la console.
resource "google_service_account" "trainer" {
  account_id   = "nba-trainer"
  display_name = "NBA MLOps Trainer"
  description  = "component=trainer : lit data/, lit et écrit models/, écrit predictions/"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "api" {
  account_id   = "nba-api"
  display_name = "NBA MLOps API"
  description  = "component=api : lit models/ et predictions/, lit la clé Anthropic"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "deployer" {
  account_id   = "nba-deployer"
  display_name = "NBA MLOps Deployer"
  description  = "component=deployer : CI/CD, pousse les images (droits de déploiement en phase 5)"

  depends_on = [google_project_service.apis]
}
