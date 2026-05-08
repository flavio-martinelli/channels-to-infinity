#=
 # Created on Wed Apr 2 2024
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces simulation data for Figure 4 and Appendix B.3.
 #              Trains networks of varying input dimension (Din), width (r),
 #              depth, and activation on the modified rosenbrock function and
 #              4 GP teachers (softplus only).
 #
 #              Optimisation rationale:
 #              - the ODEsolver is given max 1 hour to converge
 #              - the ODEsolver can stop because of abstol or reltol
 #              - the optim solver takes over and has 1 hour to converge; if after #patience iterations the loss has not decreased, the optim solver stops and we declare convergence
 #              - the optim solver can also stop because of maxtime, in this case a further finetuning is suggested
 #
=#

using DrWatson
@quickactivate

include(srcdir("loadenv.jl"))
include(srcdir("argparsing.jl"))
include(srcdir("helper_experiment.jl"))
include(srcdir("distributed.jl"))

slot_id = parsed_args["slot"]

println("Running on slot $slot_id.\nLoading processes...")

exp_name = "scaling"

settings = dict_list(Dict(:seed => seed_range(slot_id, seeds_per_slot = 10),
                          :Din => [2, 4, 8, 16],
                          :r => [2, 4, 8, 16, 
                                 @onlyif((:f == softplus), [[4,4], [4,4,4]])...],
                          :f => [softplus, sigmoid2, relu],
                          :input => standard_normal_input,
                          :teacher => [rosenbrock,
                                 @onlyif((:f == softplus && :r != [4,4] && :r != [4,4,4]), [gp_teacher1, gp_teacher2, gp_teacher3, gp_teacher4])...],
                          :biases => true))

run_experiment(exp_name, settings, n_samples_trajectory=250)
