using Documenter
using InterpolatedNyquist

makedocs(
    sitename = "InterpolatedNyquist.jl",
    format = Documenter.HTML(),
    modules = [InterpolatedNyquist],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ]
)
