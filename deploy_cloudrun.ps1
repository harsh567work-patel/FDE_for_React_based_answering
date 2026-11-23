# Google Cloud Run Deployment Script for DevCopilot
# Usage: .\deploy_cloudrun.ps1 -ProjectId "your-gcp-project-id"

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId
)

if (-not $ProjectId) {
    Write-Host ""
    Write-Host "Please provide your GCP Project ID:" -ForegroundColor Yellow
    Write-Host "  Example: .\deploy_cloudrun.ps1 -ProjectId my-assistant-project" -ForegroundColor Green
    Write-Host ""
    $ProjectId = Read-Host "Enter GCP Project ID"
}

if (-not $ProjectId) {
    Write-Host "Error: GCP Project ID is required." -ForegroundColor Red
    exit 1
}

Write-Host "==> Authenticating and setting project to: $ProjectId" -ForegroundColor Cyan
gcloud config set project $ProjectId

Write-Host "==> Deploying fde-react-assistant to Google Cloud Run (2Gi RAM, 1 CPU)..." -ForegroundColor Cyan
gcloud run deploy fde-react-assistant `
  --source . `
  --region us-central1 `
  --memory 2Gi `
  --cpu 1 `
  --max-instances 1 `
  --allow-unauthenticated `
  --env-vars-file .env.yaml

Write-Host "==> Deployment initiated. Your live URL will appear above!" -ForegroundColor Green
