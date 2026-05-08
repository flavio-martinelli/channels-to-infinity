# Teacher target functions: modified rosenbrock and 4 GP teachers (Matern32, varying length scales).

using Statistics, AbstractGPs

# --> Helpers

function compute_network(xt, f, input, Din, k; bias=true)
    if bias
        h = xt.w1[:, 1:Din] * input
        h .+= xt.w1[:, end]
        targ = xt.w2[:, 1:k] * f.(h)
        targ .+= xt.w2[:, end]
    else
        h = xt.w1 * input
        targ = xt.w2 * f.(h)
    end
    input, targ, xt
end

struct StandardizedFunction{F}
    f::F
    mean::Float64
    std::Float64
end
function StandardizedFunction(f, X)
    y = f.(eachcol(X))
    StandardizedFunction(f, mean(y), std(y))
end
(f::StandardizedFunction)(x) = (f.f(x) - f.mean) ./ f.std

# --> Teachers

struct Rosenbrock{T}
    a::Float64
    b::Float64
    c::Float64
    offset::Float64
    transform::T
    Din::Int
end
function Rosenbrock(; a = 1., b = 3., c = 1., Din = 2, transform = log10, offset = .1)
    Rosenbrock(a, b, c, offset, transform, Din)
end
function (r::Rosenbrock)(x)
    r.transform(sum(r.b * (x[i+1] + r.c - x[i]^2)^2 + (r.a - x[i])^2 for i in 1:r.Din-1) + r.offset)
end
function rosenbrock(; Nsamples, Din = 2, input = quasimontecarlo_input, kwargs...)
    X = input(; Nsamples, Din, kwargs...)
    rosen = StandardizedFunction(Rosenbrock(c=1.), X)
    X, Array(rosen.(eachcol(X))'), missing
end
function rosenbrock_unshifted(; Nsamples, Din = 2, input = quasimontecarlo_input, kwargs...)
    X = input(; Nsamples, Din, kwargs...)
    rosen = StandardizedFunction(Rosenbrock(c=0.), X)
    X, Array(rosen.(eachcol(X))'), missing
end


function gp_teacher1(; Nsamples, Din = 2, input = standard_normal_input, seed = rand(UInt), rng = Xoshiro(seed), kernel = Matern32Kernel(), kwargs...)
    X = input(; Nsamples, Din, rng, kwargs...)
    f = GP(kernel ∘ ScaleTransform(1/10))
    X, rand(rng, f(eachcol(X)))', missing
end
function gp_teacher2(; Nsamples, Din = 2, input = standard_normal_input, seed = rand(UInt), rng = Xoshiro(seed), kernel = Matern32Kernel(), kwargs...)
    X = input(; Nsamples, Din, rng, kwargs...)
    f = GP(kernel ∘ ScaleTransform(1/2))
    X, rand(rng, f(eachcol(X)))', missing
end
function gp_teacher3(; Nsamples, Din = 2, input = standard_normal_input, seed = rand(UInt), rng = Xoshiro(seed), kernel = Matern32Kernel(), kwargs...)
    X = input(; Nsamples, Din, rng, kwargs...)
    f = GP(kernel ∘ ScaleTransform(2))
    X, rand(rng, f(eachcol(X)))', missing
end
function gp_teacher4(; Nsamples, Din = 2, input = standard_normal_input, seed = rand(UInt), rng = Xoshiro(seed), kernel = Matern32Kernel(), kwargs...)
    X = input(; Nsamples, Din, rng, kwargs...)
    f = GP(kernel ∘ ScaleTransform(10))
    X, rand(rng, f(eachcol(X)))', missing
end 

