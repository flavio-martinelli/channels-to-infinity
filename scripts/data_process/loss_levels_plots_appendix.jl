#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Appendix Figure A3 (convergence diagnostics:
 #              gradient-norm and minimum-eigenvalue distributions, per network
 #              size r). Reads the same simulation outputs as loss_levels_plots.jl.
 #
 # Usage: julia loss_levels_plots_appendix.jl --bias <true|false> --f <activation>
=#
using DrWatson
@quickactivate

include(srcdir("helper_plot.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("helper_retrievers.jl"))
using PyPlot, MLPGradientFlow
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))
cmap = get_cmap("plasma", 5)

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--bias"
        help = "Specify whether to include bias (true/false)"
        arg_type = Bool
        default = false
    "--f"
        help = "Specify the function parameter (e.g., softplus)"
        arg_type = String
        default = "softplus"
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
proc_dir = mkpath(joinpath(proc_dir, suffix, "app"))  # define sub-directory for the current set of options
retrievers = [retrieve_weight_norms, retrieve_loss, retrieve_x, retrieve_gnorm, retrieve_eigs]
df = retrieve_infos(df_setup, retrievers=retrievers, verbose=true)
# <--


# --> Apply filters
df = df[df.biases .== with_bias, :]
df = df[df.f .== f, :]
sort!(df, :loss, rev=true)  # sort by loss
# <--


# --> Plot gnorm_regularized distributions
fig, axs = subplots(1, 5, figsize=(5.5,1.25), dpi=150)
rs = [1, 2, 3, 4, 5]
xlims = [df.gnorm_regularized|>minimum, df.gnorm_regularized|>maximum]
for (i, (ax, r)) in enumerate(zip(axs, rs))
    col = cmap(i-1)
    df_r = filter(row -> row.r == r, df)
    log_bins = get_log_bins(minimum(df_r.gnorm_regularized)/10,
                            maximum(df_r.gnorm_regularized)*10, 
                            25)
    line_histogram(ax, df_r.gnorm_regularized, log_bins, color="k", color_area=col)
    ax.set_xscale("log")
    ax.set_title("r = $r")
    ax.set_xlim([xlims[1]/10, xlims[2]*10])
    ax.set_xticks([1e-15, 1e-10, 1e-5])
end
fig.supxlabel("Converged gradient norm")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "gnorm_regularized.pdf"))
# <--


# --> Plot eigvals distributions
fig, axs = subplots(1, 5, figsize=(5.5,1.25), dpi=150)
rs = [1, 2, 3, 4, 5]
min_eigs = vcat([e|>minimum for e in df.eigvals]...)
xlims = [min_eigs.|>abs|>minimum, min_eigs.|>abs|>maximum]
log_bins = get_log_bins(abs(xlims[1]/10), xlims[2]*10, 25)
log_bins = [reverse(-log_bins)..., log_bins...]
for (i, (ax, r)) in enumerate(zip(axs, rs))
    col = cmap(i-1)
    df_r = filter(row -> row.r == r, df)
    line_histogram(ax, df_r.eigvals.|>minimum, log_bins, color="k", color_area=col)
    ax.set_xscale("symlog", linthresh=1e-15)
    ax.set_title("r = $r")
    ax.set_xlim([-1e-12, xlims[2]*10])
    ax.set_ylim([0, ax.get_ylim()[2]])
    ax.set_xticks([-1e-10, 0, 1e-10, 1e-4])
end
fig.supxlabel("Converged minimum eigenvalue")
fig.tight_layout(pad=0.0)
fig.savefig(joinpath(proc_dir, "eigvals.pdf"))
# <--
