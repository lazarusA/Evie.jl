using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Evie
using ExplicitImports
print_explicit_imports(Evie)
