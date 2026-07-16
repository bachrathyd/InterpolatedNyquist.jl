# Build the paper PDF (latexmk is unavailable without perl on this machine,
# so run the classic pdflatex -> bibtex -> pdflatex x2 loop explicitly).
# Usage:  powershell -File build.ps1   (from the paper/ directory)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 main.tex
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 main.tex

Write-Host "Build finished: $(Join-Path $PSScriptRoot 'main.pdf')"
