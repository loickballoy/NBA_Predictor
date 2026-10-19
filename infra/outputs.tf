output "vpc_id" {
  value = google_compute_network.vpc.id
}

output "subnet_id" {
  value = google_compute_subnetwork.main.id
}

output "bucket_name" {
  value = google_storage_bucket.mlops.name
}

output "artifact_registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "trainer_service_account" {
  value = google_service_account.trainer.email
}

output "api_service_account" {
  value = google_service_account.api.email
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}

output "trainer_job" {
  value = google_cloud_run_v2_job.trainer.name
}

output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}