# Spawns worker processes (count from --nprocs) and loads helper_experiment on each.

using Distributed

nprocs = if isdefined(Main, :parsed_args)
    get(Main.parsed_args, "nprocs", 50)
else
    50
end
@info "Adding $nprocs processes."
addprocs(nprocs, exeflags="--project=$(joinpath(@__DIR__, ".."))")

@everywhere begin
    include(joinpath(@__DIR__, "helper_experiment.jl"))
end
