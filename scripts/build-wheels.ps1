# Script to build wheels and drop the version file
param(
    [Parameter(Mandatory)]
    [string]$targetType,
    [Parameter(Mandatory)]
    [uint64]$rcVersion,
    [Parameter(Mandatory)]
    [string]$versionFilename
)

$allowedTargetTypes = @("Test", "Prod")

if( !$allowedTargetTypes.Contains($targetType) )
{
    throw "Unrecognised targetType: $targetType"
}

if( Test-Path env:FAIRLEARN_RC )
{
    throw "Environment variable FAIRLEARN_RC must not be set"
}

if( $targetType -eq "Test" )
{
    $Env:FAIRLEARN_RC = $rcVersion
}


# Store fairlearn version (including FAIRLEARN_RC) in the file
pip install .
$versionScript = Join-Path -resolve scripts fairlearn_version.py
python $versionScript > $versionFilename

# Create the packages
Write-Host
Write-Host "Creating Packages"
python setup.py sdist bdist_wheel

Write-Host
Write-Host "Package created"