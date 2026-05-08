# Input-sample generators (Gaussian, uniform cube, spherical, quasi-MC, regular 2D grid).

function standard_normal_input(; seed = rand(UInt), rng = Xoshiro(seed), Din, Nsamples, kwargs...)
    randn(rng, Din, Nsamples)
end

function standard_normal_input_fixed_seed(; seed = rand(UInt), rng = Xoshiro(seed), Din, Nsamples, kwargs...)
    randn(Xoshiro(1), Din, Nsamples)  # the seed in the argument is ignored.
end

rand_input(; Din, Nsamples, kwargs...) = rand_input(Nsamples, Din; kwargs...)

function rand_input(Nsamples, Din; seed = rand(UInt), rng = Xoshiro(seed))
    (rand(rng, Din, Nsamples) .- 0.5) * 2 * sqrt(3)
end

function uniformspherical_input(; kwargs...)
    samples = standard_normal_input(; kwargs...)
    samples ./ norm.(eachcol(samples))'
end

function quasimontecarlo_input(; Din, Nsamples, method = SobolSample(), kwargs...)
    lb = fill(-sqrt(3), Din)
    ub = fill(sqrt(3), Din)
    QuasiMonteCarlo.sample(Nsamples, lb, ub, method)
end

function regulargrid2D_input(; Din, Nsamples, kwargs...)
    grid_res = ceil(sqrt(Nsamples))
    grid_step = 2*sqrt(3)/grid_res
    x = -sqrt(3):grid_step:sqrt(3)
    y = -sqrt(3):grid_step:sqrt(3)
    X = Iterators.product(x, y) |> collect .|> collect
    hcat(X...)
end
