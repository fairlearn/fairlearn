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

Write-Host
ls
Write-Host

Write-Host "FAIRLEARN_RC = $Env:FAIRLEARN_RC"

# Set environment variable
$versionScript = Join-Path -resolve scripts fairlearn_version.py
$Env:FAIRLEARN_TEST_VERSION = python $versionScript
Write-Host "FAIRLEARN_TEST_VERSION = $Env:FAIRLEARN_TEST_VERSION"

# Create the packages
Write-Host
Write-Host "Creating Packages"
python setup.py sdist bdist_wheel

# Set the appropriate environment variable
Write-Host
Write-Host "##vso[task.setvariable variable=FAIRLEARN_TEST_VERSION;isOutput=true]$Env:FAIRLEARN_TEST_VERSION"