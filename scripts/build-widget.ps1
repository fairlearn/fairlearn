# Script to build the widget

try
{
    Push-Location fairlearn/widget/js

    # Install dependencies
    yarn install
    if ($LASTEXITCODE -ne 0)
    {
        throw "Installation of dependencies failed with result code $LASTEXITCODE"
    }

    # Do the build
    yarn build:all
    if ($LASTEXITCODE -ne 0)
    {
        throw "Build failed with result code $LASTEXITCODE"
    }

    Write-Host "Removing extra directories"
    # Tidy up
    Remove-Item -Force -Recurse dist
    Remove-Item -Force -Recurse lib
    Remove-Item -Force -Recurse node_modules
}
finally
{
    Pop-Location
}    

