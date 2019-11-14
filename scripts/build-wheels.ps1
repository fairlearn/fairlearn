# Script to build wheels and drop the version file
param(
    [Parameter(Mandatory)]
    [bool]$isTest,
    [Parameter(Mandatory)]
    [string]$versionFilename
)

if( $isTest )
{
    # Ensure that the FAIRLEARN_RC variable is set     
    Write-Host "Checking for FAIRLEARN_RC"
    if (-not (Test-Path env:FAIRLEARN_RC))
    {
        throw "Environment variable FAIRLEARN_RC not set!"
    }
    if ( [string]::IsNullOrEmpty($Env:FAIRLEARN_RC))
    {
        throw "Environment variable FAIRLEARN_RC null or empty!"
    }
    if ( [string]::IsNullOrWhiteSpace($Env:FAIRLEARN_RC))
    {
        throw "Environment variable FAIRLEARN_RC null or whitespace!"
    }
    Write-Host "FAIRLEARN_RC = $Env:FAIRLEARN_RC"
}
else
{
    # Not running for test; make sure that the FAIRLEARN_RC
    # variable is not set
    Remove-Variable $Env:FAIRLEARN_RC
}


# Set environment variable
pip install .
$versionScript = Join-Path -resolve scripts fairlearn_version.py
python $versionScript > $versionFilename

# Create the packages
Write-Host
Write-Host "Creating Packages"
python setup.py sdist bdist_wheel

Write-Host
Write-Host "Package created"