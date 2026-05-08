#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 2.
 #              Reads simulation outputs of `scripts/simulations/loss_levels.jl`
 #              and produces, for each (activation, bias) pair: loss-level
 #              clustering, finite vs. infinite-norm split, duplicate-neuron
 #              fraction, and unique-solution counts.
 #
 # Usage: julia loss_levels_plots.jl --bias <true|false> --f <activation>
=#
using DrWatson
@quickactivate

include(srcdir("helper_plot.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("helper_retrievers.jl"))
using PyPlot, MLPGradientFlow
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--bias"
        help = "Specify whether to include bias (true/false)"
        arg_type = Bool
        default = true
    "--f"
        help = "Specify the function parameter (e.g., softplus)"
        arg_type = String
        default = "g"
end;
args = parse_args(ARGS, s)


# --> Select experiment name and options
exp_name = "loss_levels"
with_bias = args["bias"]
f = eval(Meta.parse(args["f"]))
suffix = "$(f)" * (with_bias ? "_bias" : "")
# <--


# --> Extract needed information (retrievers) from the results
df_setup, sims_dir, proc_dir = setup_results(exp_name)
proc_dir = mkpath(joinpath(proc_dir, suffix))  # define sub-directory for the current set of options
retrievers = [retrieve_weight_norms, retrieve_loss, retrieve_x, 
              generate_retrieve_closest_pair("w1", cos_dist),
              generate_retrieve_closest_pair("w1", l2_dist),]
df = retrieve_infos(df_setup, retrievers=retrievers, verbose=true)
# <--


# --> Apply filters
df = df[df.biases .== with_bias, :]
df = df[df.f .== f, :]
sort!(df, :loss, rev=true)  # sort by loss
# <--


# --> Plotting loss levels
loss_lvls, idxs = cluster_loss_levels(df.loss, reltol=1e-3)

fig, ax = subplots(1,1, figsize=(2.5, 3.25), dpi=250)
loss_levels(fig, ax, df.loss, df.r, idxs, debug_mode=false)

# ticks
max_freq = 100
min_loss = minimum(df.loss)*0.8
println("min_loss = $min_loss")
ax.set_xticks([1, 90])
ax.set_xlim(ax.get_xlim()[1], 92)
ax.set_ylim(min_loss, 1)
ax.set_yticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.minorticks_off()  # Remove minor ticks
ax.spines["bottom"].set_bounds(1, 90);
ax.spines["left"].set_bounds(1.0, min_loss);
ax.spines["bottom"].set_position(("outward", 2))

fig.set_size_inches(2.5, 3.25)
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "loss_levels.pdf"))
# <--


# --> Plotting loss levels with finite/infinite solution distinction
finites = df.x .|> params .|> weightnorm .< 1e3  # 1e3 is the maxnorm defined in the experiment

# loss levels plot
fig, ax = subplots(1,1, figsize=(2.25,3.5), dpi=250)
loss_levels_binary_cat(fig, ax, df.loss, df.r, idxs, finites, ("finite", "infinite"), "Parameter norm")
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "loss_levels_inf.pdf"))

# bar plot quantifying the number of finite/infinite solutions
infinite_proportion = [1 - sum(finites[df.r .== r])/length(df.r[df.r .== r]) for r in df.r|>unique|>sort]

fig, ax = subplots(1,1, figsize=(2,1), dpi=250)
cmap = get_cmap("plasma", (unique(df.r)|>length)+1)
for r in df.r|>unique|>sort
    ax.bar(r, infinite_proportion[r], color=cmap(r), edgecolor="black", linewidth=1)
end
ax.set_xlabel("Network size")
ax.set_ylabel("% infinite")
ax.set_ylim(0, 1.05)
ax.set_xticks(df.r|>unique|>sort)
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "infinite_proportion.pdf"))
# <--


# --> Counting all duplicated solutions 
l2_thr = 1e-6  # visually chosen after plotting a few histograms: run commented lines of code below to see too
duplicates = df.min_l2_dist_w1 .< l2_thr

# for r in df.r |> unique
#     fig, ax = subplots(1,1, figsize=(3,1), dpi=250)
#     line_histogram(ax, df[df.r .== r, :min_l2_dist_w1].|>abs, get_log_bins(1e-20, 1, 100))
#     ax.set_xscale("log")
#     ax.set_ylabel(r)
#     fig.tight_layout(pad=0)
# end

# loss levels plot
fig, ax = subplots(1,1, figsize=(2.25,3.5), dpi=250)
loss_levels_binary_cat(fig, ax, df.loss, df.r, idxs, duplicates, ("duplicate", "non-duplicate"), "Solution type")
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "loss_levels_dup.pdf"))

count_duplicates = [sum(df[df.r .== r, :min_l2_dist_w1] .< l2_thr) for r in df.r|>unique|>sort]
fraction_duplicates = [sum(df[df.r .== r, :min_l2_dist_w1] .< l2_thr)/size(df[df.r .== r, :])[1] for r in df.r|>unique|>sort]
fraction_duplicates = fraction_duplicates[2:end]  # remove r=1, no duplicates can exist
println("Fraction of duplicates: $(fraction_duplicates)")

fig, ax = subplots(1,1, figsize=(1.2,1.4), dpi=250)
cmap = get_cmap("plasma", (unique(df.r)|>length))
for i in 1:length(fraction_duplicates)
    ax.bar(i+1, fraction_duplicates[i], color=cmap(i), edgecolor="black", linewidth=1)
end
ax.set_xlabel("Network size")
ax.set_title("% duplicate\nneurons")
ax.set_ylim(0, .3)
ax.set_yticks([0, .15, .30])
ax.set_yticklabels(["0%", "15%", "30%"])
ax.axhline(0.15, color="black", linestyle="--", linewidth=0.5, alpha=0.25, zorder=-1)
ax.axhline(0.30, color="black", linestyle="--", linewidth=0.5, alpha=0.25, zorder=-1)
ax.set_xlim(1.5, maximum(df.r|>unique|>sort)+0.5)
ax.set_xticks((df.r|>unique|>sort)[(df.r|>unique|>sort) .> 1])
fig.set_size_inches(1.2, 1.4)
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "fraction_duplicates.pdf"))
# <--


# --> Counting different solutions (# unique solutions computed as dissimilarities in weight space)
# very slow but worth to let it go.
unique_solutions = []
println("Computing unique solutions for each network size:")
for r in df.r |> unique
    push!(unique_solutions, compute_unique_solutions(df[df.r .== r, :x], plot_debug=true))
    println(" -> found $(unique_solutions[end]) \t unique solutions for r=$(Int(r))")
end

fig, ax = subplots(1,1, figsize=(1.2,1.4), dpi=250)
histogram_unique_solutions(fig, ax, unique_solutions, df)
fig.set_size_inches(1.2, 1.4)
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "unique_solutions.pdf"))
# <--


# --> Plotting all solutions
plot_network_grids(df, proc_dir, white_thr=0);
# <--
