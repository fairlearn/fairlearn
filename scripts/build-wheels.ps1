# Script to build wheels and drop the version file
param(
    [Parameter(Mandatory)]
    [bool]$isTest,
    [Parameter(Mandatory)]
    [uint64]$rcVersion,
    [Parameter(Mandatory)]
    [string]$versionFilename
)

if( Test-Path env:FAIRLEARN_RC )
{
    throw "Environment variable FAIRLEARN_RC must not be set"
}

if( $isTest )
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