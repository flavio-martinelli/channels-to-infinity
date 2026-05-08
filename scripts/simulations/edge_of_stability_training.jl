#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces simulation data for Figure 6 (d): SGD trajectories at
 #              several learning rates from a saddle-perturbed initialization,
 #              tracking γ and the maximum Hessian eigenvalue per iteration.
 #
 # Inputs (read from `paper_data/edge_of_stability/initialization.dat`):
 #   p   — initial parameters (NamedTuple of weights)
 #   v2  — perturbation eigenvector along the saddle's negative direction
 #   ls  — LinearSubspace describing the saddle line, used by `to_local_coords`
 #
 # The initialization file is produced by the same script that generates the
 # Figure 3 panels (saddle line + perturbation analysis); see README.
 #
 # Usage: julia edge_of_stability_training.jl <learning_rate> <channel:1|2>
 #
 # Reference learning rates explored in the paper:
 #   range_lr = [3.16, 1.0, 0.316, 0.1, 0.0316, 0.01]
 #   range_lr = [0.178, 0.0562, 0.0178]
 #   range_lr = [0.237, 0.133, 0.0750, 0.0422]
=#

using Pkg, DrWatson
@quickactivate

using MLPGradientFlow, Serialization, Optimisers, Random
import MLPGradientFlow: LinearSubspace, subspace_minloss, params, restart_ref, to_local_coords, cosine_similarity

exp_name = "edge_of_stability"

sims_dir, proc_dir = datadir("sims", exp_name), datadir("proc", exp_name)
paper_dir = projectdir("paper_data", exp_name)
mkpath(sims_dir)

function train_sgd(student, p, η, ls; kwargs...)
    res = []
    final_samples = []
    gamma = []
    max_eigval = []
    x = copy(p)
    for i in 1:500
        println("ITERATION: $i")
        _res = train(student, x; alg = Descent(η),
                     maxtime_ode = 6*60, maxiterations_ode = 10^10, save_everystep = false, maxiterations_optim = 0, kwargs...)
        x = params(_res["x"])
        println("start training final samples")
        _final_samples = train(student, x; alg = Descent(η), maxiterations_ode = 256, n_samples_trajectory = 256, maxiterations_optim = 0)
        x = params(_final_samples["x"])
        xs = params.(collect(values(_final_samples["trajectory"])))
        _gamma = first.(to_local_coords.(Ref(ls), xs))
        _max_eigval = last.(getproperty.(hessian_spectrum.(Ref(student), xs), :values))
        push!(res, _res)
        push!(final_samples, _final_samples)
        push!(gamma, _gamma)
        push!(max_eigval, _max_eigval)
        println("save results")
        serialize(joinpath(sims_dir, "results-$η-$channel.dat"),
                  (; η, gamma, max_eigval, res, final_samples))
    end
    (; η, gamma, max_eigval, res, final_samples)
end

Random.seed!(12)
input = randn(1, 2^12)
teacher = TeacherNet(; layers = ((2, sigmoid2, true), (1, identity, false)), input)
teacher.p.w1[:, 1] .= 5.
teacher.p.w1[:, 2] .= [-2.5, 2.5]
teacher.p.w2 .= 1
target = teacher(input)
student = Net(; layers = ((2, sigmoid2, false), (1, identity, false)), input, target)
p, v2, ls = deserialize(joinpath(paper_dir, "initialization.dat"))  # load initialization (committed reference)
η = parse(Float64, ARGS[1])
channel = parse(Int, ARGS[2])

res = train_sgd(student, channel == 1 ? p + 2*v2 : p, η, ls, maxtime_ode = 180, patience = 10^4, verbosity = 1)
