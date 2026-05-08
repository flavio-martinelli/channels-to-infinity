#=
 # Created on Mon Mar 10 2025
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces simulation data for Figure 2 and Appendix A.1, A.3.
 #              Trains networks with r = 1..5 hidden neurons on the modified
 #              2D rosenbrock target, with and without bias.
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

exp_name = "loss_levels"

settings = dict_list(Dict(:seed => [@onlyif((:r == 1), seed_range(slot_id, seeds_per_slot = 10))...,
                                    @onlyif((:r == 2), seed_range(slot_id, seeds_per_slot = 20))...,
                                    @onlyif((:r == 3), seed_range(slot_id, seeds_per_slot = 40))...,
                                    @onlyif((:r == 4), seed_range(slot_id, seeds_per_slot = 80))...,
                                    @onlyif((:r > 4), seed_range(slot_id, seeds_per_slot = 160))...],
                          :Din => [2],
                          :r => [1, 2, 3, 4, 5],
                          :f => g,
                          :input => [regulargrid2D_input],
                          :teacher => rosenbrock,
                          :biases => [false,true]))

run_experiment(exp_name, settings)
