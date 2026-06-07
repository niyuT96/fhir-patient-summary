param(
    [string]$BaseUrl = $env:IRIS_BASE_URL,
    [string]$Username = $env:IRIS_USERNAME,
    [string]$Password = $env:IRIS_PASSWORD
)

$ErrorActionPreference = "Stop"

if (-not $BaseUrl) {
    Write-Error "IRIS_BASE_URL is not set. Pass -BaseUrl or set it in the environment."
}

$metadataUrl = $BaseUrl.TrimEnd("/") + "/metadata"
$headers = @{
    Accept = "application/fhir+json"
}

if ($Username -and $Password) {
    $pair = "${Username}:${Password}"
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
    $basic = [Convert]::ToBase64String($bytes)
    $headers.Authorization = "Basic $basic"
}

Write-Host "Checking FHIR metadata endpoint:"
Write-Host $metadataUrl

$response = Invoke-WebRequest `
    -Uri $metadataUrl `
    -Headers $headers `
    -UseBasicParsing `
    -TimeoutSec 10

Write-Host "StatusCode: $($response.StatusCode)"
Write-Host "FHIR endpoint is reachable."
