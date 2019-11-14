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

git clone $gitURL
if($LASTEXITCODE -ne 0)
{
    throw "Error from git clone. Aborting"
}

$repoRoot = Join-Path -Resolve $baseDir fairlearn

Write-Host "repoRoot: $repoRoot"
Write-Host "Setting $targetVariable to repoRoot"
Write-Host "##vso[task.setvariable variable=$targetVariable]$repoRoot"

Set-Location $repoRoot

Write-Host "Attempting checkout of $gitCheckout"

git checkout $gitCheckout
if($LASTEXITCODE -ne 0)
{
    throw "Error from git checkout. Aborting"
}

Write-Host "Removing fairlearn subdirectory from repoRoot"
Remove-Item -Recurse -Force fairlearn

Write-Host
Get-ChildItem