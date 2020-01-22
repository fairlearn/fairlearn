# Script to set a pipeline variable from the contents of a file
# This script is only required for Azure DevOps pipelines.
param(
    [Parameter(Mandatory)]
    [string]$baseDir,
    [Parameter(Mandatory)]
    [string]$subDir,
    [Parameter(Mandatory)]
    [string]$fileName,
    [Parameter(Mandatory)]
    [string]$targetVariable
)

Write-Host $baseDir
Write-Host $subDir
Write-Host $fileName

$srcDir = Join-Path -Resolve $baseDir $subDir
$srcFile = Join-Path -Resolve $srcDir $fileName

Write-Host "Reading from $srcFile"
Write-Host "Setting variable $targetVariable"

$contents = Get-Content $srcFile

Write-Host "##vso[task.setvariable variable=$targetVariable]$contents"
Write-Host "Completed"