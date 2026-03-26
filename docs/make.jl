using Documenter
using InterpolatedNyquist

makedocs(
    sitename = "InterpolatedNyquist.jl",
    format = Documenter.HTML(),
    modules = [InterpolatedNyquist],
    checkdocs = :exports,       
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ]
)
# This is the crucial line that tells it to push to GitHub Pages!
deploydocs(
    repo = "github.com/bachrathyd/InterpolatedNyquist.jl.git",
    devbranch = "main"
)