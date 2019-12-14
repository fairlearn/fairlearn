# Script to build wheels and drop the version file
param(
    [Parameter(Mandatory)]
    [string]$targetType,
    [Parameter(Mandatory)]
    [uint64]$devVersion,
    [Parameter(Mandatory)]
    [string]$versionFilename
)

$allowedTargetTypes = @("Test", "Prod")

if( !$allowedTargetTypes.Contains($targetType) )
{
    throw "Unrecognised targetType: $targetType"
}

if( Test-Path env:FAIRLEARN_DEV_VERSION )
{
    throw "Environment variable FAIRLEARN_DEV_VERSION must not be set"
}

if( $targetType -eq "Test" )
{
    $Env:FAIRLEARN_DEV_VERSION = $devVersion
}


# Store fairlearn version (including FAIRLEARN_DEV_VERSION) in the file
Write-Host "Storing fairlearn version i $versionFilename"
pip install .
$versionScript = Join-Path -resolve scripts fairlearn_version.py
python $versionScript > $versionFilename
if ($LASTEXITCODE -ne 0)
{
    throw "fairlearn_version.py failed with result code $LASTEXITCODE"
}

# Update the ReadMe file
Write-Host
Write-Host "Updating ReadMe file"
$readMeScript = Join-Path -resolve scripts process_readme.py
$target = Join-Path -resolve $(Get-Location) README.md
python $readMeScript --input $target --output $target --loglevel INFO
if ($LASTEXITCODE -ne 0)
{
    throw "process_readme.py failed with result code $LASTEXITCODE"
}

# Create the packages
Write-Host
Write-Host "Creating Packages"
python setup.py sdist bdist_wheel

Write-Host
Write-Host "Package created"