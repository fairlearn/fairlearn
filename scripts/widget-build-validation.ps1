# Validates that the checked in files required for the widget are up to date.
# Specifically, this means that they were regenerated after changes

$widgetstaticpath = 'fairlearn/widget/static' 
$tempwidgetstaticpath = 'temp/static'

New-Item -ItemType Directory -Force -Path $tempwidgetstaticpath

Copy-Item $widgetstaticpath -Destination $tempwidgetstaticpath -Recurse

Invoke-Expression "scripts/build-widget.ps1"

Get-ChildItem $tempwidgetstaticpath | ForEach-Object {
    filename = [System.IO.Path]::GetFileName($_)
    if(Compare-Object -ReferenceObject $(Get-Content $tempwidgetstaticpath/$filename) -DifferenceObject $(Get-Content $widgetstaticpath/$filename)){
        throw "Regenerating file " + $widgetstaticpath/$filename + " changed contents. Make sure to generate the widget and check in the generated files."
    }
    Else {
        Write-Host "Regenerating file " + $widgetstaticpath/$filename + " did not change contents."
    }
}
