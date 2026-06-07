param(
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Write-Host "No .env file found. Copying .env.example to .env..."
    Copy-Item ".env.example" ".env"
    Write-Host "Edit .env before running again, especially OPENAI_API_KEY and IRIS_BASE_URL."
    exit 1
}

if ($Detached) {
    docker compose up --build -d
    Write-Host "Application started in detached mode."
    Write-Host "Open http://localhost:7860"
    Write-Host "Use 'docker compose logs -f app' to follow logs."
} else {
    docker compose up --build
}
