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

Write-Host "FAIRLEARN_RC = $Env:FAIRLEARN_RC"

# Set environment variable
pip install .
$versionScript = Join-Path -resolve scripts fairlearn_version.py
python $versionScript > test-version.txt

# Create the packages
Write-Host
Write-Host "Creating Packages"
python setup.py sdist bdist_wheel

Write-Host
Write-Host "Package created"