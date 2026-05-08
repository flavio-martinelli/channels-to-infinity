# Experiment orchestration: `run_experiment(exp_name, settings)` distributes
# parameter combinations across workers, trains each, and serializes results
# via DrWatson under `data/sims/<exp_name>/`.

using MLPGradientFlow, Distributed, DrWatson
@quickactivate
include(srcdir("helper_train.jl"))

const default_train_params = (;                            
    maxtime_ode = 2*3600,    
    maxtime_optim = 2*3600,
    patience = 10^6,
    maxiterations_ode = 10^9,
    maxiterations_optim = 10^9,
    maxnorm = 10^3,
    progress_interval = 60,
    n_samples_trajectory = 5,
    g_tol = 0.0,
    reltol = 1e-3,
    minloss = 1e-30,
    verbosity = 1,
)

function run_experiment(exp_name, settings; kwargs...)
    logdir = projectdir("logs", exp_name)
    mkpath(logdir)
    println("Starting simulations, logging will be saved in individual files in $logdir.")
    @sync @distributed for setting in settings
        dofit = true  # flag: if the simulation should be run
        rerun = false  # flag: if the simulation was already run and not converged
        filenamebase = savename(setting, allowedtypes=[Any])
        filename = datadir("sims", exp_name, filenamebase * ".jld2")
        # handle different settings
        setup_setting, train_setting = split_setting(setting)
        train_setting = merge(default_train_params, train_setting)
        train_setting = merge(train_setting, kwargs)
        net, x, xt = setup(; setup_setting...) # use DrWatson.Derived for e.g. r
        dofit, rerun, x, old_init, old_trajectory = simulation_needs_rerun(filename, dofit, rerun, x)
        if dofit
            println("Simulating $filename")
            logfile = open(joinpath(logdir, filenamebase * ".log"), "a")
            redirect_stdio(stdout = logfile, stderr = logfile) do
                @show setting

                res = train(net, x; train_setting...)

                if haskey(res, "optim_stopped_by") @show res["optim_stopped_by"] end
                if rerun
                    res["init"] = old_init
                    old_t_end = last(collect(keys(old_trajectory)))
                    for (k, v) in res["trajectory"]
                        old_trajectory[k + old_t_end] = v
                    end
                    res["trajectory"] = old_trajectory
                end
                mkpath(datadir("sims", exp_name))
                result = Dict("res" => res,
                              "setting" => setting,
                              "rerun" => rerun,
                              "train_setting" => train_setting,)
                tagsave(filename, result)
            end
            flush(logfile)
            close(logfile)
        end
    end
end

const SETUP_KWS = (:Din, :Nsamples, :seed, :rng, :parameter_rng, :teacher, :init_scale, :input, :k, :r, :f, :biases)
function split_setting(setting)
    Base.structdiff(NamedTuple(setting), NamedTuple{tuple(setdiff(keys(setting), SETUP_KWS)...)}),
    Base.structdiff(NamedTuple(setting), NamedTuple{SETUP_KWS})
end

function simulation_needs_rerun(filename, dofit, rerun, x)
    if isfile(filename) # skip simulation if already existing and converged
        res = wload(filename)
        if haskey(res["res"], "optim_stopped_by")
            stopping_criterion = res["res"]["optim_stopped_by"]
            if stopping_criterion == "patience"
                println("Already existing and converged: $filename")
                dofit = false
                return dofit, rerun, x, :nothing, :nothing
            elseif stopping_criterion == "maxtime"
                println("Already existing and not converged: $filename")
                x = params(res["res"]["x"])
                rerun = true
                old_init = res["res"]["init"]
                old_trajectory = res["res"]["trajectory"]
                return dofit, rerun, x, old_init, old_trajectory
            end
        end
    end
    dofit, rerun, x, :nothing, :nothing
end

function seed_range(slot_id = 1; seeds_per_slot = 50)
    start_seed = seeds_per_slot*(slot_id-1) + 1
    end_seed = seeds_per_slot*slot_id
    collect(start_seed:end_seed)
end

