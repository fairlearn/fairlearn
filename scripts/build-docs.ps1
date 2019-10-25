function Resolve-FullPath($path) 
{
    if (Test-Path $path)
    {
        $p = Resolve-Path $path
        return $p.Path
    }
    return $null
}

$currpath = Get-Location

try
{
    # Note that the following assume that this file is in the 'scripts'
    # subdirectory of the repository root
    $scriptpath = Resolve-FullPath (Split-Path -parent $PSCommandPath)
    $rootpath = Resolve-FullPath ([System.IO.Path]::Combine($scriptpath, ".."))

    $codepath = [System.IO.Path]::Combine($rootpath, "fairlearn")
    $docbuildpath = [System.IO.Path]::Combine($rootpath, "docbuild")
    $docconfigpath = [System.IO.Path]::Combine($rootpath, "docs")

    # Make sure we have a clean slate
    Remove-Item -Path $docbuildpath -Recurse -Force -ErrorAction SilentlyContinue

    # Make sure we are running from the repository root
    Set-Location -Path $rootpath

    # Copy the doc configurations to the build path
    Copy-Item $docconfigpath -Destination "$docbuildpath" -Recurse -Force

    # Move into the docbuild directory
    Set-Location -Path $docbuildpath

    # Create some expected directories
    New-Item "_static" -ItemType Directory -Force
    New-Item "_build" -ItemType Directory -Force
    New-Item "_templates" -ItemType Directory -Force

    Write-Host "Building API doc"
    & sphinx-apidoc "$codepath" -o "$docbuildpath"

    Write-Host "Building Docs" 
    & sphinx-build . _build
}
finally
{
    Set-Location -Path $currpath
}
