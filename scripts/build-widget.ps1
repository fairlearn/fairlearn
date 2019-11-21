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

    # Tidy up
    Remove-Item -Recurse dist
    Remove-Item -Recurse lib
    Remove-Item -Recurse node_modules
}
finally
{
    Pop-Location
}    

