#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 4 and the related Appendix B
 #              series (channel frequency and properties across architectures
 #              and teacher datasets). Reads simulation outputs of
 #              `scripts/simulations/scaling.jl`.
 #
 # Usage: julia scaling_plots.jl --f <activation>
=#
using DrWatson
@quickactivate

include(srcdir("helper_plot.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("helper_retrievers.jl"))
include(srcdir("colors.jl"))
using PyPlot, MLPGradientFlow, StatsBase
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))
cmap = get_cmap("plasma", 6)

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--f"
        help = "Specify the function parameter (e.g., softplus)"
        arg_type = String
        default = "softplus"
end;
args = parse_args(ARGS, s)


# --> Select experiment name and options
exp_name = "scaling"
if args["f"] == "all"
    f = nothing
else
    f = eval(Meta.parse(args["f"]))
end
suffix = args["f"]
# <--

# --> Extract needed information (retrievers) from the results
df_setup, sims_dir, proc_dir = setup_results(exp_name)
retrievers = [retrieve_weight_norms, retrieve_loss, retrieve_x, retrieve_gnorm,
              generate_retrieve_closest_pair("w1", cos_dist),
              generate_retrieve_closest_pair("w1", l2_dist),
              generate_retrieve_closest_pair("w2", cos_dist),
              generate_retrieve_closest_pair("w3", cos_dist),
              generare_retrieve_closest_triplet("w1", cos_dist),
              generate_retrieve_cosine_tuplets_below_thr("w1", 1e-2),
              generate_retrieve_trajectory(cut_idx=250),
              generate_retrieve_x_from_trajectory(cut_idx=250),
              generate_retrieve_gnorm_from_trajectory(cut_idx=250),
              generate_retrieve_wnorm_from_trajectory(cut_idx=250)]  # cut trajectory at 250 for unexpected simulation re-starts
df = retrieve_infos(df_setup, retrievers=retrievers, verbose=true)
# <--


# --> Apply filters
df = df[df.f .== f, :]
suffix = suffix * "_filt"
proc_dir = mkpath(joinpath(proc_dir, suffix))  # define sub-directory for the current set of options
# <--

# --> Apply filters
df = df[df.biases .== with_bias, :]
df = df[df.f .== f, :]
sort!(df, :loss, rev=true)  # sort by loss
# <--

# distance between x and x_trj 
df.dist_x_x_trj = [norm(params(df.x[i]) - params(df.x_trj[i])) for i in 1:size(df, 1)]
# compute angle between x and x_trj
df.angle_x_x_trj = [cos_dist(params(df.x[i]), params(df.x_trj[i])) for i in 1:size(df, 1)]

# plot line histogram with log bins of the distance between x and x_trj
bins = get_log_bins(minimum(abs.(df.angle_x_x_trj) .+ 1e-16), maximum(df.angle_x_x_trj), 50)
fig, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=250)
line_histogram(ax, abs.(df.angle_x_x_trj) .+ 1e-16, bins; color_area=nothing, color="black")
ax.set_xscale("log")
ax.set_xlim(1e-17, 1)
ax.set_xticks([1e-16, 1e-8, 1e-4, 1e-2, 1e-1, 1])
fig.tight_layout(pad=0)
fig.savefig(joinpath(proc_dir, "angle_x_x_trj.pdf"))

# for safety, one can remove the uncertain simulations where the angle is larger than 1e-4.
# it will remove about 10% of the simulations. It avoids considering simulations that jumped too far when switching from ode
# solvers to optimizer to reach a minimum.

# --> remove uncertain simulations
df = df[df.angle_x_x_trj .< 1e-2, :]
# <--


# --> plot wnorm distributions only for r = [4,4]
fig, ax = subplots(1, 1, figsize=(8,1), dpi=250)
r = [4,4]
col = cmap(4)
df_r = filter(row -> row.r == r, df)
# log_bins = get_log_bins(minimum(df_r.wnorm)/10,
#                         maximum(df_r.wnorm)*10, 
#                         25)
scalar_bins = range(0, stop=1005, length=300)
line_histogram(ax, df_r.wnorm, scalar_bins, color="k", color_area=col)
ax.set_yscale("log")
# ax.set_ylabel(r)
ax.set_xlim([0, 1005])
ax.set_xticks([0, 50, 100, 200, 400, 600, 800, 1000])
fig.tight_layout(pad=0.1)


# --> plot gnorm_regularized distributions
fig, axs = subplots(3, 2, figsize=(5.5/2,3), dpi=150)
rs = [2, 4, 8, 16, [4,4], [4,4,4]]
xlims = [df.gnorm_trj|>minimum, df.gnorm_trj|>maximum]
for (i, (ax, r)) in enumerate(zip(axs, rs))
    col = cmap(i-1)
    df_r = filter(row -> row.r == r, df)
    log_bins = get_log_bins(minimum(df_r.gnorm_trj)/10,
                            maximum(df_r.gnorm_trj)*10, 
                            25)
    line_histogram(ax, df_r.gnorm_trj, log_bins, color="k", color_area=col)
    ax.set_xscale("log")
    ax.set_ylabel(r)
    ax.set_xlim([xlims[1]/10, xlims[2]*10])
    ax.set_xticks([1e-15, 1e-10, 1e-5])
end
fig.supxlabel("Converged gradient norm")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "gnorm_trj.pdf"))
# <--


# --> plot wnorm distributions
fig, axs = subplots(3, 2, figsize=(5.5/2,3), dpi=150)
rs = [2, 4, 8, 16, [4,4], [4,4,4]]
xlims = [df.wnorm_trj|>minimum, df.wnorm_trj|>maximum]
for (i, (ax, r)) in enumerate(zip(axs, rs))
    col = cmap(i-1)
    df_r = filter(row -> row.r == r, df)
    log_bins = get_log_bins(minimum(df_r.wnorm_trj)/10,
                            maximum(df_r.wnorm_trj)*10, 
                            25)
    line_histogram(ax, df_r.wnorm_trj, log_bins, color="k", color_area=col)
    ax.set_xscale("log")
    ax.set_ylabel(r)
    ax.set_xlim([xlims[1]/10, xlims[2]*10])
    # ax.set_yscale("log")
end
fig.supxlabel("Converged weight norm")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "wnorm_trj.pdf"))
# <--


# --> frequency of maxnorm minima
maxnorm = 1e3*1
max_angle = 1e-2
# plot stats on frequency of minima of higher than maxnorm for various settings
# change in r
freq = Dict()
confidence_intervals = Dict()
for (k, subdf) in pairs(groupby(df, :r))
    freq[string(k.r)] = mean(subdf.wnorm_trj .> maxnorm)
    confidence_intervals[string(k.r)] = binomial_CI(subdf.wnorm_trj .> maxnorm)
end
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
rs = f == softplus ? ["2", "4", "8", "16", "[4, 4]", "[4, 4, 4]"] : ["2", "4", "8", "16"]
vals = [freq[r] for r in rs]
fig, ax = count_plot(fig, ax, rs, vals.*100, 
                     formatter = x -> @sprintf("%d", x),
                     confidence_intervals = confidence_intervals)
ax.set_ylabel("% maxnorm minima")
ax.set_xlabel("hidden layer sizes")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "maxnorm_minima_per_r.pdf"))
# change in d_in
freq = Dict()
confidence_intervals = Dict()
for (k, subdf) in pairs(groupby(df, :Din))
    freq[string(k.Din)] = mean(subdf.wnorm_trj .> maxnorm)
    confidence_intervals[string(k.Din)] = binomial_CI(subdf.wnorm_trj .> maxnorm)
end
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)    
dins = ["2", "4", "8", "16"]
vals = [freq[d] for d in dins]
fig, ax = count_plot(fig, ax, dins, vals.*100, 
                     formatter = x -> @sprintf("%d", x),
                     confidence_intervals = confidence_intervals)
ax.set_ylabel("% maxnorm minima")
ax.set_xlabel("input dimension")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "maxnorm_minima_per_din.pdf"))
# change in teacher
freq = Dict()
confidence_intervals = Dict()
for (k, subdf) in pairs(groupby(df, :teacher))
    freq[string(k.teacher)] = mean(subdf.wnorm_trj .> maxnorm)
    confidence_intervals[string(k.teacher)] = binomial_CI(subdf.wnorm_trj .> maxnorm)
end
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
teachers = f == softplus ? ["rosenbrock", "gp_teacher1", "gp_teacher2", "gp_teacher3", "gp_teacher4"] : ["rosenbrock"]
vals = [freq[t] for t in teachers]
fig, ax = count_plot(fig, ax, teachers, vals.*100,
                     formatter = x -> @sprintf("%d", x),
                     confidence_intervals = confidence_intervals)
ax.set_ylabel("% maxnorm minima")
ax.set_xlabel("dataset")
ax.set_xticklabels(f == softplus ? ["rosenbrock", "GP (s = 0.1)", "GP (s = 0.5)", "GP (s = 2.0)", "GP (s = 10.0)"] : ["rosenbrock"], rotation=45)
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "maxnorm_minima_per_dataset.pdf"))
# change in teacher but considering also close pair of neurons
freq = Dict()
confidence_intervals = Dict()
for (k, subdf) in pairs(groupby(df, :teacher))
    mask = subdf.wnorm_trj .> maxnorm .&& subdf.min_cos_dist_w1 .< max_angle
    freq[string(k.teacher)] = mean(mask)
    confidence_intervals[string(k.teacher)] = binomial_CI(mask)
end
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
teachers = f == softplus ? ["rosenbrock", "gp_teacher1", "gp_teacher2", "gp_teacher3", "gp_teacher4"] : ["rosenbrock"]
vals = [freq[t] for t in teachers]
fig, ax = count_plot(fig, ax, teachers, vals.*100,
                     formatter = x -> @sprintf("%d", x),
                     confidence_intervals = confidence_intervals)
ax.set_ylabel("% channels")
ax.set_ylim([0, 100])
ax.set_xlabel("dataset")
ax.set_xticklabels(f == softplus ? ["rosenbrock", "GP (s = 0.1)", "GP (s = 0.5)", "GP (s = 2.0)", "GP (s = 10.0)"] : ["rosenbrock"], rotation=45)
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "channel_minima_per_dataset.pdf"))
# <--


# --> Comparing losses of finite and maxnorm minima
### TODO: for the moment the losses are not computed from the trajectory, but from the final converged point.
# if we do not filter the simulations, we must use the trajectory AND recompute the loss. I skip this for now.
losses_finite = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] .< maxnorm]
losses_maxnorm = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] > maxnorm]
losses_channels = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] > maxnorm && df.min_cos_dist_w1[i] .< max_angle]

# plot overlapping, normalized line histograms with log x bins
log_bins = get_log_bins(minimum(df.ode_loss), maximum(df.ode_loss), 50)
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=150)
line_histogram(ax, losses_finite, log_bins, color="red", normalize=true, label="finite", linewidth=0.75)
line_histogram(ax, losses_channels, log_bins, color="blue", normalize=true, label="channel", linewidth=0.75)
# line_histogram(ax, losses_maxnorm, log_bins, color="red", normalize=true, label="maxnorm", linewidth=0.75)
ax.set_xscale("log")
uniquefy_legend(ax, loc="upper left", fontsize=8)
ax.set_xlabel("loss")
ax.set_ylabel("density")
fig.tight_layout(pad=0.0)   
fig.savefig(joinpath(proc_dir, "losses_distribution.pdf"))


### IMPORTANT NOTE: the above plot suffers from very high difference in frequency of channels vs. finite minima.
# For GPs with large s, the finite are very rare, but normalization is done across all teachers, this results in over-expressing
# the effect of large s GPs. Need to plot teacher by teacher as done below.

# compute same thing but for different teachers
for teacher in df.teacher|>unique
    losses_finite = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] .< maxnorm && df.teacher[i] == teacher]
    losses_maxnorm = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] > maxnorm && df.teacher[i] == teacher]
    losses_channels = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] > maxnorm && df.min_cos_dist_w1[i] .< max_angle && df.teacher[i] == teacher]
    # plot overlapping, normalized line histograms with log x bins
    log_bins = get_log_bins(minimum(vcat([losses_channels..., losses_finite...])),
                        maximum(vcat([losses_channels..., losses_finite...])), 25)
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
    line_histogram(ax, losses_finite, log_bins, color="red", normalize=true, label="finite", linewidth=0.75)
    line_histogram(ax, losses_channels, log_bins, color="blue", normalize=true, label="channel", linewidth=0.75)
    ax.set_xscale("log")
    ax.set_xlim([minimum(log_bins)/1.05, maximum(log_bins)*1.05])
    ax.set_xlabel("loss")
    ax.set_ylabel("density")
    uniquefy_legend(ax, loc="upper left", fontsize=7)    
    fig.tight_layout(pad=0.0)
    fig.savefig(joinpath(proc_dir, "losses_distribution_$(teacher).pdf"))
end

# redo plot for gp_teacher2
losses_finite = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] .< maxnorm && df.teacher[i] == gp_teacher2]
losses_channels = [df.ode_loss[i] for i in 1:size(df, 1) if df.wnorm_trj[i] > maxnorm && df.min_cos_dist_w1[i] .< max_angle && df.teacher[i] == gp_teacher2]
log_bins = get_log_bins(minimum(vcat([losses_channels..., losses_finite...])),
                        maximum(vcat([losses_channels..., losses_finite...])), 25)
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=150)
line_histogram(ax, losses_finite, log_bins, color="red", normalize=true, label="finite", linewidth=0.75)
line_histogram(ax, losses_channels, log_bins, color="blue", normalize=true, label="channel", linewidth=0.75)
ax.set_xscale("log")
ax.set_xlim([1.5e-3, 2])
ax.set_yticks([0, 0.1])
ax.set_xlabel("loss")
ax.set_ylabel("density")
uniquefy_legend(ax, loc="upper left", fontsize=7)
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "losses_distribution_gp_teacher2.pdf"))

# --> Plotting the angle between the closest pair of weights
cmap = get_cmap("plasma", 4)

df_finite = df[df.wnorm_trj .< maxnorm, :]
df_infinite = df[df.wnorm_trj .> maxnorm, :]

for teacher in unique(df_infinite.teacher)
    df_t = filter(row -> row.teacher == teacher, df_infinite)

    fig, axs = subplots(2, 2, figsize=(5.5,3), dpi=150)
    rs = [2, 4, 8, 16]
    for (i, (ax, r)) in enumerate(zip(axs, rs))
        col = i<5 ? cmap(i-1) : (0,0,0,0.5)
        df_r = filter(row -> row.r == r, df_t)
        # compute strength of the pair of output weights involved
        a_sum = map(row -> abs(row.x["w2"][row.pair_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.pair_cos_dist_w1[2]]), 
                        eachrow(df_r))
        # check if the pair of output weights involved have opposite signs
        a_opp = map(row -> row.x["w2"][row.pair_cos_dist_w1[1]]*
                            row.x["w2"][row.pair_cos_dist_w1[2]] < 0, 
                            eachrow(df_r))
        ax.plot(df_r.min_cos_dist_w1[a_opp .== 1], a_sum[a_opp .== 1], color=col, linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
        ax.plot(df_r.min_cos_dist_w1[a_opp .== 0], a_sum[a_opp .== 0], color=col, linestyle="", marker="x", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
        ax.set_xscale("symlog", linthresh=1e-7)
        ax.set_xlim([-5e-8, 5])
        ax.set_xticks([0, 1e-7, 1e-4, 1e-1])
        ax.set_yscale("log")
        ax.set_ylim([5e-1, 3e3])
        # if i < 3 ax.set_ylabel(L"$|a_1| + |a_2|$") end
        ax.set_title("r = $r")
        ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
        # set text at top right of the above line
        ax.text(0.015, 1.3e3, " > 8°", ha="left", va="top", fontsize=8)
    end
    fig.legend([L"$a_1 \cdot a_2 < 0$", L"$a_1 \cdot a_2 \geq 0$"], loc="center", fontsize=8, handletextpad=0.0, bbox_to_anchor=(0.5, 0.53))
    fig.supxlabel(L"Closest pair distance $d(w_1, w_2) = 1 -  \frac{w_1 \cdot w_2}{||w_1|| \  ||w_2||}$")
    fig.supylabel(L"$|a_1| + |a_2|$")
    fig.tight_layout(pad=0)
    # fig.savefig(joinpath(proc_dir, "angle_w1_w2_$(teacher).pdf"))
    # close(fig)
end
# <--

# --> Plotting the angle between the closest triplets of weights
rs = [2, 4, 8, 16]
for teacher in unique(df_infinite.teacher)
    fig, axs = subplots(2, 2, figsize=(5.5,3), dpi=150)
    df_t = filter(row -> row.teacher == teacher, df_infinite)
    for (i, (ax, r)) in enumerate(zip(axs, rs))
        if r == 2 continue end
        col = i<5 ? cmap(i-1) : (0,0,0,0.5)
        df_r = filter(row -> row.r == r, df_t)
        # compute strength of the pair of output weights involved
        a_sum = map(row -> abs(row.x["w2"][row.triplet_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[2]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[3]]), 
                        eachrow(df_r))
        ax.plot(df_r.min_triplet_cos_dist_w1, a_sum, color=col, linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
        ax.set_xscale("symlog", linthresh=1e-7)
        ax.set_xlim([-5e-8, 5])
        ax.set_xticks([0, 1e-7, 1e-4, 1e-1])
        ax.set_yscale("log")
        ax.set_ylim([5e-1, 3e3])
        # if i < 3 ax.set_ylabel(L"$|a_1| + |a_2|$") end
        ax.set_title("r = $r")
        ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
        # set text at top right of the above line
        ax.text(0.015, 1.3e3, " > 8°", ha="left", va="top", fontsize=8)
    end
    fig.supxlabel(L"Closest triplet distance $d(w_1, w_2, w_3) = 1 -  \frac{w_1 \cdot w_2 + w_1 \cdot w_3 + w_2 \cdot w_3}{||w_1|| \  ||w_2|| \ ||w_3||}$")
    fig.supylabel(L"$|a_1| + |a_2| + |a_3|$")
    fig.tight_layout(pad=0.4)
    fig.savefig(joinpath(proc_dir, "angle_w1_w2_w3_$(teacher).pdf"))
end


# --> Plot of angles and output weights

df_infinite = df
for teacher in unique(df_infinite.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)

    fig, ax = subplots(1, 1, figsize=(2,1.5), dpi=150)
    # compute strength of the pair of output weights involved
    a_sum = map(row -> abs(row.x["w2"][row.pair_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.pair_cos_dist_w1[2]]), 
                        eachrow(df_dataset))
    # check if the pair of output weights involved have opposite signs
    a_opp = map(row -> row.x["w2"][row.pair_cos_dist_w1[1]]*
                        row.x["w2"][row.pair_cos_dist_w1[2]] < 0, 
                        eachrow(df_dataset))
    ax.plot(df_dataset.min_cos_dist_w1[a_opp .== 1], a_sum[a_opp .== 1], color="blue", linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.plot(df_dataset.min_cos_dist_w1[a_opp .== 0], a_sum[a_opp .== 0], color="blue", linestyle="", marker="x", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlim([-5e-8, maximum(df_dataset.min_cos_dist_w1)*1.5])
    ax.set_xticks([0, 1e-7, 1e-4, 1e-2, 1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 3e3])
    ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
    # set text at top right of the above line
    ax.text(0.015, 2e3, " > 8°", ha="left", va="top", fontsize=8)
    ax.set_xlabel(L"min$_{ij} \ d(w_i, w_j)$")
    ax.set_ylabel(L"$|a_i| + |a_j|$")
    ax.legend([L"$a_i \cdot a_j < 0$", L"$a_i \cdot a_j \geq 0$"], loc="lower left", fontsize=8, handletextpad=0.0)
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "all_angle_w1_w2_$(string(teacher)).pdf"))

    # same but for triplets (skip r=2)
    df_dataset = filter(row -> row.teacher == teacher && row.r != 2, df_dataset)
    fig, ax = subplots(1, 1, figsize=(2,1.5), dpi=150)
    # compute strength of the triplet of output weights involved
    a_sum = map(row -> abs(row.x["w2"][row.triplet_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[2]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[3]]), 
                        eachrow(df_dataset))
    ax.plot(df_dataset.min_triplet_cos_dist_w1, a_sum, color="blue", linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim([-5e-8, maximum(df_dataset.min_cos_dist_w1)*1.5])
    ax.set_xticks([0, 1e-7, 1e-4, 1e-2, 1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 3e3])
    ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
    # set text at top right of the above line
    ax.text(0.015, 2e3, " > 8°", ha="left", va="top", fontsize=8)
    ax.set_xlabel(L"min$_{ijk} \ d(w_i, w_j, w_k)$")
    ax.set_ylabel(L"$|a_i| + |a_j| + |a_k|$")
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "all_angle_w1_w2_w3_$(string(teacher))_triplets.pdf"))
    close(fig)

    # plot |a1| + |a2|  vs 1/min_pair_l2_dist_w1
    fig, ax = subplots(1, 1, figsize=(2,1.5), dpi=150)
    # compute strength of the pair of output weights involved
    a_sum = map(row -> abs(row.x["w2"][row.pair_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.pair_cos_dist_w1[2]]), 
                        eachrow(df_dataset))
    # check if the pair of output weights involved have opposite signs
    a_opp = map(row -> row.x["w2"][row.pair_cos_dist_w1[1]]*
                        row.x["w2"][row.pair_cos_dist_w1[2]] < 0, 
                        eachrow(df_dataset))
    ax.plot(1 ./ df_dataset.min_l2_dist_w1[a_opp .== 1], a_sum[a_opp .== 1], color="blue", linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.plot(1 ./ df_dataset.min_l2_dist_w1[a_opp .== 0], a_sum[a_opp .== 0], color="blue", linestyle="", marker="x", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.set_xscale("symlog", linthresh=1e-7)
    # ax.set_xlim([-5e-8, 5])
    # ax.set_xticks([0, 1e-7, 1e-4, 1e-1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 3e3])
    ax.set_xlabel(L"$1/||w_i - w_j||$")
    ax.set_ylabel(L"$|a_i| + |a_j|$")
    ax.legend([L"$a_i \cdot a_j < 0$", L"$a_i \cdot a_j \geq 0$"], loc="lower right", fontsize=8, handletextpad=0.0)
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "all_angle_w1_w2_1_min_l2_dist_$(string(teacher)).pdf"))
    close(fig)
end


df_infinite = df[df.wnorm_trj .> maxnorm, :]
for teacher in unique(df_infinite.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)

    fig, ax = subplots(1, 1, figsize=(2.5,2), dpi=150)
    # compute strength of the pair of output weights involved
    a_sum = map(row -> abs(row.x["w2"][row.pair_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.pair_cos_dist_w1[2]]), 
                        eachrow(df_dataset))
    # check if the pair of output weights involved have opposite signs
    a_opp = map(row -> row.x["w2"][row.pair_cos_dist_w1[1]]*
                        row.x["w2"][row.pair_cos_dist_w1[2]] < 0, 
                        eachrow(df_dataset))
    ax.plot(df_dataset.min_cos_dist_w1[a_opp .== 1], a_sum[a_opp .== 1], color="blue", linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.plot(df_dataset.min_cos_dist_w1[a_opp .== 0], a_sum[a_opp .== 0], color="blue", linestyle="", marker="x", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlim([-5e-8, 5])
    ax.set_xticks([0, 1e-7, 1e-4, 1e-1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 3e3])
    ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
    # set text at top right of the above line
    ax.text(0.015, 2e3, " > 8°", ha="left", va="top", fontsize=8)
    ax.set_xlabel("Input weight distance\nof closest pair")
    ax.set_ylabel(L"$|a_1| + |a_2|$")
    ax.legend([L"$a_1 \cdot a_2 < 0$", L"$a_1 \cdot a_2 \geq 0$"], loc="lower left", fontsize=8, handletextpad=0.0)
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "maxnorm_angle_w1_w2_$(string(teacher)).pdf"))

    # same but for triplets (skip r=2)
    df_dataset = filter(row -> row.teacher == teacher && row.r != 2, df_dataset)
    fig, ax = subplots(1, 1, figsize=(2.5,2), dpi=150)
    # compute strength of the triplet of output weights involved
    a_sum = map(row -> abs(row.x["w2"][row.triplet_cos_dist_w1[1]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[2]])+
                        abs(row.x["w2"][row.triplet_cos_dist_w1[3]]), 
                        eachrow(df_dataset))
    ax.plot(df_dataset.min_triplet_cos_dist_w1, a_sum, color="blue", linestyle="", marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="k")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim([5e-5, 5])
    ax.set_xticks([0, 1e-4, 1e-1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 3e3])
    ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
    # set text at top right of the above line
    ax.text(0.015, 2e3, " > 8°", ha="left", va="top", fontsize=8)
    ax.set_xlabel("Input weight distance\nof closest triplet")
    ax.set_ylabel(L"$|a_1| + |a_2| + |a_3|$")
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "maxnorm_angle_w1_w2_w3_$(string(teacher))_triplets.pdf"))
end
# <--

# --> Check distance between Δw and w for the pair of input weights belonging to channels
function angle_w_Δw(x, pair)
    w = x.w1[pair[2], :]
    Δw = x.w1[pair[1], :] - w
    ϵ = norm(Δw)
    Δw ./= ϵ
    abs(1 - cos_dist(Δw, w))
end

for teacher in unique(df_infinite.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)
    channels = filter(row -> row.min_cos_dist_w1 .< 1e-2, df_dataset)

    dists = [angle_w_Δw(row.x|>params, row.pair_cos_dist_w1) for row in eachrow(channels)]

    # plot histogram of dists
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1), dpi=150)
    bins = 0:0.01:1
    line_histogram(ax, dists, bins, color="black", color_area=nothing)
    ax.set_xlabel(L"cos_sim($\Delta w$, $w$)")
    ax.set_ylabel("count")
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "hist_angle_w_Δw_$(string(teacher)).pdf"))

    # same histogram but with log bins and 1 - cos_dist
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1), dpi=150)
    log_bins = get_log_bins(minimum(1 .- dists), maximum(1 .- dists), 50)
    line_histogram(ax, 1 .- dists, log_bins, color="black", color_area=nothing)
    ax.set_xscale("log")
    ax.set_xlabel(L"1 - cos_sim($\Delta w$, $w$)")
    ax.set_ylabel("count")
    fig.tight_layout(pad=0)
    fig.savefig(joinpath(proc_dir, "hist_angle_w_Δw_log_$(string(teacher)).pdf"))

    # scatter plot of dists vs min_cos_dist_w1
    # fig, ax = plt.subplots(1, 1, figsize=(2.5, 3), dpi=150)
    # ax.plot(channels.min_cos_dist_w1, dists, color="blue", linestyle="", marker="o",
    # markersize=4, markeredgewidth=0.5, markeredgecolor="k", alpha = 0.5)
    # ax.set_xscale("symlog", linthresh=1e-7)
    # ax.set_xlim([-5e-8, 5])
    # ax.set_xticks([0, 1e-7, 1e-4, 1e-1])
    # ax.set_ylim([0, 1.1])
    # ax.axvline(0.01, color="k", linestyle="--", linewidth=0.5)
    # # set text at top right of the above line
    # ax.text(0.015, 0.5, " > 8°", ha="left", va="top", fontsize=8)
    # ax.set_xlabel("Input weight distance\nof closest pair")
    # ax.set_ylabel(L"cos_sim($\Delta w$, $w$)")
    # fig.tight_layout(pad=0)
    # fig.savefig(joinpath(proc_dir, "angle_w_Δw_vs_min_cos_dist_$(string(teacher)).pdf"))
end 
# <--

# --> compute angle between Δx and pair of output weights
function angle_a_Δx(trj, tuplets; skip=1)
    all_indices = unique(vcat(tuplets...))
    alignments = []
    times = []
    for i in 1:length(trj)-(1+skip)  # avoid potential last step of optim (does not follow ODE)
        t_ode0, x0 = trj[i]
        t_ode1, x1 = trj[i+skip]
        Δx = params(x1) - params(x0)  # update vector 
        Δx_proj = Δx*0.0  # project Δx onto the subspace spanned by the tuplets
        Δx_proj.w2[all_indices] = Δx.w2[all_indices]
        push!(alignments, norm(Δx_proj) / norm(Δx))  # how much of the update is in the direction of the subspace
        push!(times, t_ode1)
    end
    alignments, times
end

function fraction_a_Δx(trj, tuplets; skip=1)
    all_indices = unique(vcat(tuplets...))
    alignments = []
    times = []
    for i in 1:length(trj)-(1+skip)  # avoid potential last step of optim (does not follow ODE)
        t_ode0, x0 = trj[i]
        t_ode1, x1 = trj[i+skip]
        Δx = params(x1) - params(x0)  # update of parameter vector 
        x0 = params(x0)
        # compute average w weight per channel (tuplet) and concatenate into a matrix
        W_avg = reduce(hcat, [mean(x0.w1[tup, :], dims=1)[:] for tup in tuplets])
        Pw = W_avg * (W_avg'*W_avg)^-1 * W_avg'  # projection operator onto the subspace spanned by the input weights of the channels
        Δx_proj_w1 = Δx.w1 * Pw  # project Δx onto the subspace spanned by the tuplets
        # output weights are easier, just erase all components of the update that are not in the tuplets
        Δx_proj_w2 = Δx.w2 * 0.0
        Δx_proj_w2[all_indices] = Δx.w2[all_indices]
        # merge Δx_proj_w1 and Δx_proj_w2 into a single vector
        Δx_proj = Δx * 0.0
        Δx_proj.w1 = Δx_proj_w1
        Δx_proj.w2 = Δx_proj_w2
        push!(alignments, norm(Δx_proj) / norm(Δx))  # how much of the update is in the direction of the subspace
        push!(times, t_ode1)
    end
    alignments, times
end

for teacher in unique(df_infinite.teacher)
    println("Processing teacher: $teacher")
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)
    channels = filter(row -> row.min_cos_dist_w1 .< 1e-3, df_dataset)

    alignments = []
    times = []
    for row in eachrow(channels)
        tuplets = row.cos_tuplets_below_thr_w1
        # (a, t) = angle_a_Δx(row.trajectory, tuplets, skip=1)  
        (a, t) = fraction_a_Δx(row.trajectory, tuplets, skip=1)  
        push!(alignments, a)
        push!(times, t)
        print(".")
    end

    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=250)
    # histogram of average last 50 elements of alignments
    bins = 0:0.02:1
    average_last_50 = [mean(a[end-50:end]) for a in alignments]
    first_updates = [a[1] for a in alignments]
    line_histogram(ax, [a[1] for a in alignments], bins, color="grey", color_area=nothing, label="early training", normalize=true)
    line_histogram(ax, average_last_50, bins, color="blue", color_area=nothing, label="late training", normalize=true)
    ax.set_xlabel(L"$\frac{||\Delta x_{\Gamma}||}{||\Delta x||}$")
    ax.set_ylabel("frequency")
    ax.set_yticks([0, 0.3, 0.6])
    ax.set_yticklabels(["0%", "30%", "60%"])
    uniquefy_legend(ax, loc="upper left", fontsize=7)
    # ax.set_yscale("log")
    fig.tight_layout(pad=0.1)
    # fig.savefig(joinpath(proc_dir, "histogram_Δx_Γ_$(string(teacher)).pdf"))
end

# same plot but only on pairs of weights given by the closest pair
for teacher in unique(df_infinite.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)
    channels = filter(row -> row.min_cos_dist_w1 .< 1e-2, df_dataset)
    channels_single_pair = filter(row -> length(row.cos_tuplets_below_thr_w1) == 1, channels)

    alignments = []
    times = []
    for row in eachrow(channels_single_pair)
        tuplets = row.pair_cos_dist_w1
        (a, t) = angle_a_Δx(row.trajectory, tuplets)
        push!(alignments, a)
        push!(times, t)
        print(".")
    end
    fig, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=250)
    # histogram of average last 50 elements of alignments
    bins = 0:0.02:1
    average_last_50 = [mean(a[end-50:end]) for a in alignments]
    first_updates = [a[1] for a in alignments]
    line_histogram(ax, [a[1] for a in alignments], bins, color="grey", color_area=nothing, label="early training", normalize=true)
    line_histogram(ax, average_last_50, bins, color="blue", color_area=nothing, label="late training", normalize=true)
    ax.set_xlabel("Fraction of update\nparallel to saddle line", fontsize=7)
    ax.set_ylabel("frequency")
    ax.set_yticks([0, 0.3, 0.6])
    ax.set_yticklabels(["0%", "30%", "60%"])
    uniquefy_legend(ax, loc="upper left", fontsize=7)
    fig.tight_layout(pad=0.1)
    fig.savefig(joinpath(proc_dir, "histogram_Δx_Γ_single_pair_$(string(teacher)).pdf"))
end

alignments_per_dataset = []
for teacher in unique(df_infinite.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df_infinite)
    channels = filter(row -> row.min_cos_dist_w1 .< 1e-3, df_dataset)

    alignments = []
    times = []
    for row in eachrow(channels)
        tuplets = row.cos_tuplets_below_thr_w1
        # (a, t) = angle_a_Δx(row.trajectory, tuplets)
        (a, t) = fraction_a_Δx(row.trajectory, tuplets, skip=1)  
        push!(alignments, a)
        push!(times, t)
        print(".")
    end
    fig, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=250)
    # histogram of average last 50 elements of alignments
    bins = 0:0.02:1
    average_last_50 = [mean(a[end-50:end]) for a in alignments]
    first_updates = [a[1] for a in alignments]
    line_histogram(ax, [a[1] for a in alignments], bins, color="grey", color_area=nothing, label="early training", normalize=true)
    line_histogram(ax, average_last_50, bins, color="blue", color_area=nothing, label="late training", normalize=true)
    ax.set_xlabel("Fraction of update\nparallel to saddle line", fontsize=7)
    ax.set_ylabel("frequency")
    ax.set_yticks([0, 0.3, 0.6])
    ax.set_yticklabels(["0%", "30%", "60%"])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    uniquefy_legend(ax, loc="upper left", fontsize=7)
    fig.tight_layout(pad=0.1)
    fig.savefig(joinpath(proc_dir, "histogram_Δx_Γ_single_pair_$(string(teacher)).pdf"))
    push!(alignments_per_dataset, (average_last_50, teacher))
end

# replot gp_teacher2
df_dataset= filter(row -> row.teacher == gp_teacher2, df_infinite)
channels = filter(row -> row.min_cos_dist_w1 .< 1e-3, df_dataset)

alignments = []
times = []
for row in eachrow(channels)
    tuplets = row.cos_tuplets_below_thr_w1
    # (a, t) = angle_a_Δx(row.trajectory, tuplets)
    (a, t) = fraction_a_Δx(row.trajectory, tuplets, skip=1)  
    push!(alignments, a)
    push!(times, t)
    print(".")
end
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=250)
# histogram of average last 50 elements of alignments
bins = 0:0.02:1
average_last_50 = [mean(a[end-50:end]) for a in alignments]
first_updates = [a[1] for a in alignments]
line_histogram(ax, [a[1] for a in alignments], bins, color="grey", color_area=nothing, label="early training", normalize=true)
line_histogram(ax, average_last_50, bins, color="blue", color_area=nothing, label="late training", normalize=true)
ax.set_xlabel("Fraction of update\nparallel to saddle line", fontsize=7)
ax.set_ylabel("frequency")
ax.set_yticks([0, 0.3, 0.6])
ax.set_yticklabels(["0%", "30%", "60%"])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
uniquefy_legend(ax, loc="upper left", fontsize=6.5)
fig.tight_layout(pad=0.1)
fig.savefig(joinpath(proc_dir, "histogram_Δx_Γ_single_pair_teacher2_fig.pdf"))



frac_by_dataset = Dict(dataset => sum(vals .> 0.9) / length(vals)
                       for (vals, dataset) in alignments_per_dataset)
# counting channels per dataset but now removing the ones not updating parallel to the saddle
maxnorm = 1e3*1
max_angle = 1e-3
freq = Dict()
confidence_intervals = Dict()
for (k, subdf) in pairs(groupby(df, :teacher))
    mask = subdf.wnorm_trj .> maxnorm .&& subdf.min_cos_dist_w1 .< max_angle
    freq[string(k.teacher)] = mean(mask)*frac_by_dataset[k.teacher]
    confidence_intervals[string(k.teacher)] = binomial_CI(freq[string(k.teacher)], length(mask))
end
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
teachers = f == softplus ? ["rosenbrock", "gp_teacher1", "gp_teacher2", "gp_teacher3", "gp_teacher4"] : ["rosenbrock"]
vals = [freq[t] for t in teachers]
fig, ax = count_plot(fig, ax, teachers, vals.*100,
                     formatter = x -> @sprintf("%d", x),
                     confidence_intervals = confidence_intervals)
ax.set_ylabel("% channels")
ax.set_ylim([0, 100])
ax.set_xlabel("dataset")
ax.set_xticklabels(f == softplus ? ["rosenbrock", "GP (s = 0.1)", "GP (s = 0.5)", "GP (s = 2.0)", "GP (s = 10.0)"] : ["rosenbrock"], rotation=45)
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "channel_minima_per_dataset.pdf"))


# --> quantify how many pairs, triplets etc... are present in the channels

tuplet_types = []
for teacher in unique(df.teacher)
    df_dataset= filter(row -> row.teacher == teacher, df)
    channels = filter(row -> row.min_cos_dist_w1 .< 1e-2, df_dataset)
    # count lenght of tuplets
    tuplet_lengths = [[length(tup) for tup in row.cos_tuplets_below_thr_w1] 
         for row in eachrow(channels)]
    tuplet_lengths = vcat(tuplet_lengths...)
    # count each occurence
    push!(tuplet_types, countmap(tuplet_lengths))
end


all_tuplet_sizes = sort(unique(vcat([collect(keys(d)) for d in tuplet_types]...)))
n_datasets = length(tuplet_types)
bar_width = 1.2
group_width = bar_width * maximum(length.(tuplet_types)) + 0.1  # space between datasets
# Compute x positions for each bar
x_positions = []
x_labels = []
for (i, dict) in enumerate(tuplet_types)
    sizes = sort(collect(keys(dict)))
    for (j, size) in enumerate(sizes)
        xpos = (i-1)*group_width + j*bar_width
        push!(x_positions, xpos)
        push!(x_labels, "$(size-1)")
    end
end
heights = vcat([ [dict[size] for size in sort(collect(keys(dict)))] for dict in tuplet_types ]...)

fig, ax = plt.subplots(1, 1, figsize=(2.5,1.5), dpi=200)
ax.bar(x_positions, heights, width=bar_width, color="C0", edgecolor="k", linewidth=0.7)
ax.set_yscale("log")
ax.set_ylabel("Count")
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)
ax.set_title("Dimensionality of channels")
fig.tight_layout(pad=0, rect=(0, 0.15, 1, 0.975))

teacher_string = ["GP (s = 0.1)", "GP (s = 0.5)", "GP (s = 2.0)", "GP (s = 10.0)", "rosenbrock"]
# Add teacher names below each group
for (i, teacher) in enumerate(unique(df.teacher))
    # Find the x positions for this group
    dict = tuplet_types[i]
    n_bars = length(dict)
    # Compute the center x position of the group
    group_start = (i-1)*group_width
    group_end = group_start + (n_bars-1)*bar_width
    x_center = (group_start + group_end + 1) / 2
    ax.text(x_center, 0.25*ax.get_ylim()[1], teacher_string[i],
            ha="center", va="top", fontsize=6, rotation=15, color="k", clip_on=false)
end

fig.savefig(joinpath(proc_dir, "tuplet_type_frequencies_per_dataset.pdf"))

# <--

# --> Compute hessian max eig along trajectory only for channel solutions and gp_teacher2
teacher = gp_teacher2
df_dataset= filter(row -> row.teacher == teacher, df_infinite)
channels = filter(row -> row.min_cos_dist_w1 .< 1e-2, df_dataset)

# find back rows of df_setup matching with channels. Use: r, din, seed, f, teacher
df_setup_rich = retrieve_infos(df_setup, retrievers=[], verbose=true)
df_setup_full = hcat(df_setup, df_setup_rich, makeunique=true)
cols = [:r, :Din, :seed, :f, :teacher]
channels_with_res = innerjoin(channels, df_setup_full, on=cols, makeunique=true)

max_eigvals_list_channels = []
min_eigvals_list_channels = []
ode_time_channels = []
norms_channels = []
idxs = unique(round.(Int, exp.(range(log(1), log(length(channels_with_res.trajectory[1])), length=50))))
for row in eachrow(channels_with_res)
    net, _ = netfromres(row.res)
    # span few points across the trajectory in log space
    xs = [row.trajectory[i][2] for i in idxs]
    norm_w2s = [weightnorm(x|>params) for x in xs]
    evals = [eigen(hessian(net, x|>params)).values for x in xs]
    max_eigvals = [e|>maximum for e in evals]
    min_eigvals = [e|>minimum for e in evals]
    min_pos_eigvals = [e[e .> 0.0]|>minimum for e in evals]
    ode_time = [row.trajectory[i][1] for i in idxs]
    push!(max_eigvals_list_channels, max_eigvals)
    push!(min_eigvals_list_channels, min_eigvals)
    push!(ode_time_channels, ode_time)
    push!(norms_channels, norm_w2s)
    print(".")
end

# same computation but for the finite minima
df_finite = df[df.wnorm_trj .< maxnorm/2, :]
df_dataset = filter(row -> row.teacher == teacher, df_finite)
finite_with_res = innerjoin(df_dataset, df_setup_full, on=cols, makeunique=true)
max_eigvals_list_finite = []
min_eigvals_list_finite = []
ode_time_finite = []
for row in eachrow(finite_with_res)
    net, _ = netfromres(row.res)
    # span few points across the trajectory in log space
    xs = [row.trajectory[i][2] for i in idxs]
    evals = [eigen(hessian(net, x|>params)).values for x in xs]
    max_eigvals = [e|>maximum for e in evals]
    min_eigvals = [e|>minimum for e in evals]
    min_pos_eigvals = [e[e .> 0.0]|>minimum for e in evals]
    ode_time = [row.trajectory[i][1] for i in idxs]
    push!(max_eigvals_list_finite, max_eigvals)
    push!(min_eigvals_list_finite, min_eigvals)
    push!(ode_time_finite, ode_time)
    print(".")
end

fig, ax = plt.subplots(1, 1, figsize=(3.5/2, 1), dpi=350)
for (max_e, ode_t) in zip(max_eigvals_list_finite, ode_time_finite)
    ax.plot(ode_t, max_e, color="red", alpha = 0.05, label="finite")
end
for (max_e, ode_t, norms) in zip(max_eigvals_list_channels, ode_time_channels, norms_channels)
    stop_idx = length(norms[norms .< maxnorm])
    ax.plot(ode_t[1:stop_idx], max_e[1:stop_idx], color="blue", alpha = 0.05, label="channels")
end
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(L"$t_{ODE}$")
ax.set_ylabel(L"\lambda_{max}")
ax.set_xlim([1e-2, 1e10])
ax.set_xticks([1e-2, 1e2, 1e6, 1e10])
ax.set_yticks([1e1, 1e5])
leg = uniquefy_legend(ax, loc="upper left", fontsize=5)
for line in leg.get_lines()
    line.set_linewidth(1); line.set_alpha(0.5)
end
fig.tight_layout(pad=0.15)
fig.savefig(joinpath(proc_dir, "hessian_max_eigvals_$(string(teacher)).pdf"))

# same but for min_eigvals
fig, ax = plt.subplots(1, 1, figsize=(3.5/2, 1), dpi=350)
for (min_e, ode_t) in zip(min_eigvals_list_finite, ode_time_finite)
    ax.plot(ode_t, min_e, color="red", alpha = 0.05, label="finite")
end
for (min_e, ode_t) in zip(min_eigvals_list_channels, ode_time_channels)
    ax.plot(ode_t, min_e, color="blue", alpha = 0.05, label="channels")
end
ax.set_yscale("symlog", linthresh=1e-7)
ax.set_yticks([-1e-2, -1e-6, 1e-6, 1e-2])
ax.set_xscale("log")
ax.set_xlabel(L"$t_{ODE}$")
ax.set_ylabel(L"\lambda_{min}")
ax.set_xlim([1e-2, 1e10])
ax.set_ylim([-1e-1, 1e-1])
for line in leg.get_lines()
    line.set_linewidth(1); line.set_alpha(0.5)
end
fig.tight_layout(pad=0.15)
fig.savefig(joinpath(proc_dir, "hessian_min_eigvals_$(string(teacher)).pdf"))
# <--



# --> Loss distribution for finite and infinite minima with new test set
teacher = rosenbrock
df_dataset= filter(row -> row.teacher == teacher, df_infinite)
channels = filter(row -> row.min_cos_dist_w1 .< 1e-2, df_dataset)
# find back rows of df_setup matching with channels. Use: r, din, seed, f, teacher
df_setup_rich = retrieve_infos(df_setup, retrievers=[], verbose=true)
df_setup_full = hcat(df_setup, df_setup_rich, makeunique=true)
cols = [:r, :Din, :seed, :f, :teacher]
channels_with_res = innerjoin(channels, df_setup_full, on=cols, makeunique=true)
df_finite = df[df.wnorm_trj .< maxnorm/2, :]
df_dataset = filter(row -> row.teacher == teacher, df_finite)
finite_with_res = innerjoin(df_dataset, df_setup_full, on=cols, makeunique=true)

test_losses_channels = []
train_losses_channels = []
rs_channels = []
for row in eachrow(channels_with_res)
    if row.r|>length >1 continue  end
    net, _ = netfromres(row.res)
    inp, targ, xt = teacher(; input=standard_normal_input, Din=row.Din, Nsamples=10^3, parameter_rng=Xoshiro(123456), rng=Xoshiro(123456))
    net = Net(layers = ((row.r, f, true), (1, identity, true)),
              input = inp, target = targ, derivs = 2)
    test_loss = loss(net, row.x|>params)
    push!(test_losses_channels, test_loss)
    push!(train_losses_channels, row.loss)
    push!(rs_channels, row.r)
    println("Loss $(row.loss) -- Test loss $(test_loss)")
end

test_losses_finite = []
train_losses_finite = []
rs_finite = []
for row in eachrow(finite_with_res)
    if row.r|>length >1 continue  end
    net, _ = netfromres(row.res)
    inp, targ, xt = teacher(; input=standard_normal_input, Din=row.Din, Nsamples=10^3, parameter_rng=Xoshiro(123456), rng=Xoshiro(123456))
    net = Net(layers = ((row.r, f, true), (1, identity, true)),
              input = inp, target = targ, derivs = 2)
    test_loss = loss(net, row.x|>params)
    push!(test_losses_finite, test_loss)
    push!(train_losses_finite, row.loss)
    push!(rs_finite, row.r)
    println("Loss $(row.loss) -- Test loss $(test_loss)")
end

# plot overlapping, normalized line histograms with log x bins
for r in [2, 4, 8, 16]
    losses_channels = test_losses_channels[rs_channels .== r]
    losses_finite = test_losses_finite[rs_finite .== r]
    log_bins = get_log_bins(minimum(losses_channels), maximum(losses_finite), 50)
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=150)
    line_histogram(ax, losses_finite, log_bins, color="red", normalize=true, label="finite", linewidth=0.75)
    line_histogram(ax, losses_channels, log_bins, color="blue", normalize=true, label="channel", linewidth=0.75)
    # line_histogram(ax, losses_maxnorm, log_bins, color="red", normalize=true, label="maxnorm", linewidth=0.75)
    ax.set_xscale("log")
    uniquefy_legend(ax, loc="upper left", fontsize=8)
    ax.set_xlabel("loss")
    ax.set_ylabel("density")
    fig.tight_layout(pad=0.0) 
    println("Losses for r = $r:")
    println("Mean finite train loss: $(round(mean(train_losses_finite[rs_finite .== r]), digits=4))")
    println("Mean finite loss: $(round(mean(losses_finite), digits=4))")
    println("Mean channel train loss: $(round(mean(train_losses_channels[rs_channels .== r]), digits=4))")
    println("Mean channel loss: $(round(mean(losses_channels), digits=4))")
end
# <--

# --> Frequency of channel minima in deep nets
df_r = filter(row -> row.r == [4,4,4], df_infinite)
n = size(df_r)[1]

# assigning a new column (per layer) to df_infinite that categorises the minima
assignment_col_names = ["plateau_inf_w1", "plateau_inf_w2", "plateau_inf_w3"]
for (col, bottom_layer, top_layer) in zip(assignment_col_names, ["w1", "w2", "w3"], ["w2", "w3", "w4"])
    # skip assignemnt if r != [4,4,4]
    df_r[!, col] = map(row ->row.r == [4,4,4] && row.wnorm_trj .> maxnorm &&
                        row["min_cos_dist_$(bottom_layer)"] < 1e-2,
                        eachrow(df_r))
end

freq_w1 = mean(df_r.plateau_inf_w1 .& .!df_r.plateau_inf_w2 .& .!df_r.plateau_inf_w3)
freq_w2 = mean(df_r.plateau_inf_w2 .& .!df_r.plateau_inf_w1 .& .!df_r.plateau_inf_w3)
freq_w3 = mean(df_r.plateau_inf_w3 .& .!df_r.plateau_inf_w1 .& .!df_r.plateau_inf_w2)
freq_w1_w2 = mean(df_r.plateau_inf_w1 .& df_r.plateau_inf_w2 .& .!df_r.plateau_inf_w3)
freq_w2_w3 = mean(df_r.plateau_inf_w2 .& df_r.plateau_inf_w3 .& .!df_r.plateau_inf_w1)
freq_w1_w3 = mean(df_r.plateau_inf_w1 .& df_r.plateau_inf_w3 .& .!df_r.plateau_inf_w2)
freq_w1_w2_w3 = mean(df_r.plateau_inf_w1 .& df_r.plateau_inf_w2 .& df_r.plateau_inf_w3)
freq_empty = mean(.!df_r.plateau_inf_w1 .& .!df_r.plateau_inf_w2 .& .!df_r.plateau_inf_w3)

labels = ["L1", "L1&L2", "L2", "L2&L3", "L3", "L1&L3"]
frequencies = [freq_w1, freq_w1_w2, freq_w2, freq_w2_w3, freq_w3, freq_w1_w3]

label_colors = Dict(
    labels[3] => COLOR_MAP3_PASTEL[0b100],
    labels[4] => COLOR_MAP3_PASTEL[0b110],
    labels[5] => COLOR_MAP3_PASTEL[0b010],
    labels[6] => COLOR_MAP3_PASTEL[0b011],
    labels[1] => COLOR_MAP3_PASTEL[0b001],
    labels[2] => COLOR_MAP3_PASTEL[0b101]
)

fig = figure(figsize=(3,2), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  # 3:1 ratio = 75%:25%
ax = fig.add_subplot(gs[1], projection="polar")
ax2 = fig.add_subplot(gs[2])

N = length(frequencies)
angles = mod.(collect(range(0, -2π, length=N+1)[1:end-1]) .+ π/2, 2π)  
base_radius = 2
# Create bars with bases starting at base_radius
bars = ax.bar(angles, frequencies .* 100, width=2π/1.5N, bottom=base_radius, linewidth=0.5, edgecolor="k", zorder=3)
for (bar, label) in zip(bars, labels)
    bar.set_facecolor(label_colors[label])
end
# Add labels
ax.set_yticks([])
ax.spines["polar"].set_visible(false)
ax.set_ylim([0, 21])
# Show the plot
ax.set_xticks([])
for (angle, label) in zip(angles, labels)
    ax.text(angle, 16, label, ha="center", va="center", fontsize=8, rotation=0)
end
ax.yaxis.set_tick_params(color="lightgray", labelsize=6, which="major")
ax.grid(false)
for (angle, freq) in zip(angles, frequencies)
    ax.text(angle, base_radius + freq*100 + 1.5, @sprintf("%d%%", freq*100),
        ha="center", fontsize=6, color="k", zorder=10)
end

ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines["top"].set_visible(false)
ax2.spines["bottom"].set_visible(false)
ax2.spines["left"].set_visible(false)
ax2.spines["right"].set_visible(false)
ax_inset = ax2.inset_axes([0.0, 0.42, 1, 0.25])  # [x, y, width, height] in axes coordinates
ax_inset.bar(1:2, [freq_w1_w2_w3, freq_empty] .* 100, color=[(1,1,1), COLOR_MAP3_PASTEL[0b111]], edgecolor="k", linewidth=0.5)
ax_inset.set_xticks([1, 2])
ax_inset.set_xticklabels(["L1&L2&L3", "None"], rotation=35)
# ax_inset.set_ylim([0, 35])
ax_inset.set_yticks([])
ax_inset.set_yticklabels([])
# Add percentage text on top of each bar
ax_inset.text(1, freq_w1_w2_w3*100 + 2, @sprintf("%d%%", round(Int, freq_w1_w2_w3*100)),
    ha="center", fontsize=6, color="k", zorder=10)
ax_inset.text(2, freq_empty*100 + 2, @sprintf("%d%%", round(Int, freq_empty*100)),
    ha="center", fontsize=6, color="k", zorder=10)
ax_inset.yaxis.set_tick_params(labelsize=7) 

fig.tight_layout(pad=0, h_pad=0, w_pad=0)
fig.savefig(joinpath(proc_dir, "radial_plot_r444.pdf"))
# <--



# --> Plot stats on number simulated results
# simulations per teacher
counts = combine(groupby(df, :teacher), nrow => :count)
counts.teacher = string.(counts.teacher)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.teacher, counts.count)
ax.set_title("# runs per dataset")
fig.savefig(joinpath(proc_dir, "simulations_per_teacher.pdf"))
# other axes but exclude gp_teacher1, gp_teacher2, gp_teacher3, gp_teacher4
# simulations per d_in
exclude_teachers = ["gp_teacher1", "gp_teacher2", "gp_teacher3", "gp_teacher4"]
df_filtered = filter(row -> !(string(row.teacher) in exclude_teachers), df)
counts = combine(groupby(df_filtered, :Din), nrow => :count)
counts.Din = string.(counts.Din|>sort)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.Din, counts.count)
ax.set_title("# runs per d_in (rosenbrock)")
fig.savefig(joinpath(proc_dir, "simulations_per_din.pdf"))
# simulations per r
counts = combine(groupby(df_filtered, :r), nrow => :count)
counts.r = string.(counts.r)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.r, counts.count)
ax.set_title("# runs per r (rosenbrock)")
fig.savefig(joinpath(proc_dir, "simulations_per_r.pdf"))
# simulations per f
counts = combine(groupby(df_filtered, :f), nrow => :count)
counts.f = string.(counts.f)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.f, counts.count)
ax.set_title("# runs per activation fn.\n(rosenbrock)")
fig.savefig(joinpath(proc_dir, "simulations_per_f.pdf"))
# now only the gp_teachers
exclude_teachers = ["rosenbrock"]
df_filtered = filter(row -> !(string(row.teacher) in exclude_teachers), df)
counts = combine(groupby(df_filtered, :Din), nrow => :count)
counts.Din = string.(counts.Din|>sort)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.Din, counts.count)
ax.set_title("# runs per d_in (gp)")
fig.savefig(joinpath(proc_dir, "simulations_per_din_gp.pdf"))
# simulations per r
counts = combine(groupby(df_filtered, :r), nrow => :count)
counts.r = string.(counts.r|>sort)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=150)
fig, ax = count_plot(fig, ax, counts.r, counts.count)
ax.set_title("# runs per r (gp)")
fig.savefig(joinpath(proc_dir, "simulations_per_r_gp.pdf"))
# <--

# --> plot a sigmoid and its derivative
x = range(-6, 6, length=100)
y_softplus = softplus.(x)
y = 1 ./ (1 .+ exp.(-x))
y_prime = y .* (1 .- y)

fig, ax = plt.subplots(1, 1, figsize=(1/4, 1/4), dpi=150)
ax.plot(x, y, color="black", linewidth=1.5)
ax.axis("off")
fig.set_size_inches(1/4, 1/4)
fig.savefig(joinpath(proc_dir, "sigmoid.pdf"))

fig, ax = plt.subplots(1, 1, figsize=(1/4, 1/4), dpi=150)
ax.plot(x, y_prime, color="black", linewidth=1.5)
ax.axis("off")
fig.set_size_inches(1/4, 1/4)
fig.savefig(joinpath(proc_dir, "sigmoid_prime.pdf"))

fig, ax = plt.subplots(1, 1, figsize=(1/4, 1/4), dpi=150)
ax.plot(x, y_softplus, color="black", linewidth=1.5)
ax.axis("off")
fig.set_size_inches(1/4, 1/4)
fig.savefig(joinpath(proc_dir, "softplus.pdf"))
# <--
