# Ensures the project environment is instantiated and loads MLPGradientFlow.

using Pkg
if !Pkg.Operations.is_instantiated(Pkg.Types.EnvCache())
    Pkg.instantiate()
end
using MLPGradientFlow