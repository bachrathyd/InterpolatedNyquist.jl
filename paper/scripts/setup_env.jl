# One-time environment setup for the paper scripts.
#   julia paper/scripts/setup_env.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(dirname(@__DIR__)))
Pkg.add(["CairoMakie", "MDBM", "OrdinaryDiffEq", "QuadGK", "ForwardDiff",
         "StaticArrays", "BenchmarkTools", "DelimitedFiles"])
Pkg.instantiate()
Pkg.precompile()
