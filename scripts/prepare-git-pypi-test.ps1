# Script to make a clone of fairlearn at the specified tag in a given directory
# Writes the path to the repo root out as a VSO variable

param(
    [Parameter(Mandatory)]
    [string]$gitURL,
    [Parameter(Mandatory)]
    [string]$baseDir,
    [Parameter(Mandatory)]
    [string]$targetVariable,
    [string]$gitCheckout = "master"
)

Set-Location $baseDir

git checkout $gitURL

$repoRoot = Join-Path -Resolve $baseDir fairlearn

Write-Host "repoRoot: $repoRoot"
Write-Host "Setting $targetVariable to repoRoot"
Write-Host "##vso[task.setvariable variable=$targetVariable]$repoRoot"

Set-Location $repoRoot

WriteHost "Attempting checkout of $gitCheckout"

git checkout $gitCheckout