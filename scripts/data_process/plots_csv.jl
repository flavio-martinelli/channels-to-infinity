#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 5 (`all_curves.pdf`) and
 #              Figure 6 (a)-(c) (`function1.pdf`, `function2.pdf`,
 #              `loss_landscape_trj.pdf`, `loss_at_infinity.pdf`).
 #
 # NOTE: This script consumes pre-computed CSV files placed under
 #       `data/sims/plots_csv/`. The script that produces those CSVs is not
 #       yet included in the repo; it will be added by the author. Expected
 #       inputs and column schemas:
 #
 #   function.csv             — x, target, saddleline, studentone, studenttwo
 #   loss_at_infinity.csv     — w, loss
 #   loss_at_infinity_optima.csv — w, loss   (3 rows: saddle, channel 1, channel 2)
 #   losslandscape.csv        — gamma, proj, loss   (regular grid)
 #   projected_dynamics.csv   — flow_{i1,i2,1,2,sgd1,sgd2,adam1,adam2}_{gamma,proj},
 #                              saddle_{gamma,proj}
 #   losscurve.csv            — x, y   (loss - loss_inf vs 1/eps)
 #   ccurve.csv, acurve.csv   — x, y   (gating coeffs vs 1/eps)
 #   simcurve.csv             — x, y   (cos(Delta, w) vs 1/eps)
 #   approxcurve.csv          — x, y   (approximation error vs 1/eps)
 #   maxeigvals.csv           — x, y   (max Hessian eigval vs 1/eps)
=#
using DrWatson
@quickactivate

include(srcdir("helper_plot.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("helper_retrievers.jl"))
include(srcdir("colors.jl"))
using PyPlot, MLPGradientFlow, StatsBase
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

exp_name = "plots_csv"
sims_dir = projectdir("paper_data", exp_name)  # committed reference CSVs (see README)
proc_dir = datadir("proc", exp_name)
mkpath(proc_dir)

# Load the xy data
using CSV, DataFrames
function_df = CSV.read(joinpath(sims_dir, "function.csv"), DataFrame)
loss_at_infinity_df = CSV.read(joinpath(sims_dir, "loss_at_infinity.csv"), DataFrame)
loss_at_infinity_optima_df = CSV.read(joinpath(sims_dir, "loss_at_infinity_optima.csv"), DataFrame)

figsize = (2, 1.5)
fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=250, sharey=true)
ax.plot(function_df.x, function_df.target, color="blue", label="target", alpha=1, linewidth=0.25)
ax.plot(function_df.x, function_df.studenttwo, color="darkorange", label="channel 2", alpha=1, linewidth=0.25)

ax.set_xlabel(L"$x$")
ax.set_ylabel(L"$y$")
ax.legend(loc="upper left", fontsize=6, frameon=false)
fig.tight_layout(pad=0)
fig.set_size_inches(figsize)
fig.savefig(joinpath(proc_dir, "function1.pdf"))

fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=250, sharey=true)
ax.plot(function_df.x, function_df.saddleline, color="red", label="saddle line", alpha=1, linewidth=0.25)
ax.plot(function_df.x, function_df.studentone, color="green", label="channel 1", alpha=1, linewidth=0.25)
ax.set_xlabel(L"$x$")
ax.set_ylabel(L"$y$")
ax.legend(loc="upper left", fontsize=6, frameon=false)
fig.tight_layout(pad=0)
fig.set_size_inches(figsize)
fig.savefig(joinpath(proc_dir, "function2.pdf"))


# load the loss landscape and trajectories
LogNorm = PyPlot.matplotlib.colors.LogNorm
FormatStrFormatter = PyPlot.matplotlib.ticker.FormatStrFormatter

loss_df = CSV.read(joinpath(sims_dir, "losslandscape.csv"), DataFrame)
trj_df = CSV.read(joinpath(sims_dir, "projected_dynamics.csv"), DataFrame)

fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=250)
gamma_vals = unique(loss_df.gamma)
proj_vals = unique(loss_df.proj)
Z = reshape(loss_df.loss, (length(gamma_vals), length(proj_vals)))
minval = minimum(Z)  # Smallest positive value
maxval = maximum(Z)
levels = exp.(range(log(minval), log(maxval), length=50))

c = ax.contourf(gamma_vals, proj_vals, Z', levels=levels, cmap="viridis", norm=LogNorm())
ax.set_xlim(minimum(gamma_vals), maximum(gamma_vals))
ax.set_ylim(minimum(proj_vals), maximum(proj_vals))

# plot trajectories
ax.plot(trj_df.flow_i1_gamma, trj_df.flow_i1_proj, color="white", linewidth=0.5, label="infinite data")
ax.plot(trj_df.flow_1_gamma, trj_df.flow_1_proj, color="grey", linewidth=0.5, label="finite data")
ax.plot(trj_df.flow_sgd1_gamma, trj_df.flow_sgd1_proj, color="orange", linewidth=0.5, label="SGD")
ax.plot(trj_df.flow_adam1_gamma, trj_df.flow_adam1_proj, color="red", linewidth=0.5, label="Adam")

ax.plot(trj_df.flow_i2_gamma, trj_df.flow_i2_proj, color="white", linewidth=0.5, label="infinite data")
ax.plot(trj_df.flow_2_gamma, trj_df.flow_2_proj, color="grey", linewidth=0.5, label="finite data")
ax.plot(trj_df.flow_sgd2_gamma, trj_df.flow_sgd2_proj, color="orange", linewidth=0.5, label="SGD")
ax.plot(trj_df.flow_adam2_gamma, trj_df.flow_adam2_proj, color="red", linewidth=0.5, label="Adam")

ax.plot(trj_df.saddle_gamma, trj_df.saddle_proj, color="white", linewidth=0.5, label="infinite data")

cb = fig.colorbar(c, ax=ax, label="loss", ticks=[1e-1, 1e-2, 1e-3])
ax.set_xlabel(L"$\gamma$")
ax.set_ylabel(L"$\alpha$")

uniquefy_legend(ax, fontsize=6, loc="upper right", frameon=true)

ax.spines["right"].set_visible(true)
ax.spines["top"].set_visible(true)

fig.tight_layout(pad=0)
fig.set_size_inches(3, 1.5)
fig.savefig(joinpath(proc_dir, "loss_landscape_trj.pdf"))


# plot the loss at infinity
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=250)
ax.plot(loss_at_infinity_df.w, loss_at_infinity_df.loss, color="black", alpha=1, linewidth=1)
ax.scatter(loss_at_infinity_optima_df.w[1], loss_at_infinity_optima_df.loss[1], color="red", s=4, zorder=10, label="saddle line")
ax.scatter(loss_at_infinity_optima_df.w[2], loss_at_infinity_optima_df.loss[2], color="green", s=4, zorder=10, label="channel 1")
ax.scatter(loss_at_infinity_optima_df.w[3], loss_at_infinity_optima_df.loss[3], color="orange", s=4, zorder=10, label="channel 2")
ax.set_xlabel(L"$w$")
ax.set_ylabel("loss")
ax.legend(loc="upper right", fontsize=6, frameon=false, handletextpad=0.1)
# color area between x =[0.7993, 3.2022]
fig.tight_layout(pad=0)
fig.set_size_inches(1.5, 1.5)
ax.fill_betweenx(ax.get_ylim(), 0.7993, 3.2022, color="blue", alpha=0.075, zorder=-1, edgecolor="none", linewidth=0)
ax.set_ylim(0.0, 0.062)
fig.savefig(joinpath(proc_dir, "loss_at_infinity.pdf"))

# Load other data :    
# - losscurve.csv
# - ccurve.csv
# - acurve.csv
# - simcurve.csv
# - approxcurve.csv

losscurve = CSV.read(joinpath(sims_dir, "losscurve.csv"), DataFrame)
ccurve = CSV.read(joinpath(sims_dir, "ccurve.csv"), DataFrame)
acurve = CSV.read(joinpath(sims_dir, "acurve.csv"), DataFrame)
simcurve = CSV.read(joinpath(sims_dir, "simcurve.csv"), DataFrame)
approxcurve = CSV.read(joinpath(sims_dir, "approxcurve.csv"), DataFrame)
maxeigcurve = CSV.read(joinpath(sims_dir, "maxeigvals.csv"), DataFrame)

fig, axs = plt.subplots(2, 3, figsize=(5.5, 1.75), dpi=250)

axs[1,1].scatter(ccurve.x, ccurve.y, color="blue", s=8)
axs[1,1].set_ylabel(L"$c$")
axs[1,2].scatter(acurve.x, acurve.y, color="blue", s=8)
axs[1,2].set_ylabel(L"$a$")
axs[1,3].scatter(simcurve.x, simcurve.y, color="blue", s=8)
axs[1,3].set_ylabel(L"$\cos(\Delta, w)$")
axs[2,1].scatter(losscurve.x, losscurve.y, color="blue", s=8)
axs[2,1].set_ylabel(L"loss - loss$_\infty$")
axs[2,1].set_yscale("log")
axs[2,2].scatter(approxcurve.x, approxcurve.y, color="blue", s=8)
axs[2,2].set_ylabel("approximation\nerror")
axs[2,2].set_yscale("log")
axs[2,3].scatter(maxeigcurve.x, maxeigcurve.y, color="blue", s=8)
axs[2,3].set_ylabel(L"$\lambda_{max}$")
axs[2,3].set_yscale("log")

# draw x=1/y^2 in the first plot for a range of 1 to 1e4
x_vals = range(minimum(approxcurve.x), maximum(approxcurve.x), length=100)
y_vals = 1 ./ (x_vals.^2) * 2e-1
axs[2,2].plot(x_vals, y_vals, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
# add text 1/e^2 at the right of the top left part of line
axs[2,2].text(minimum(approxcurve.x)*5, y_vals[1]/10, L"$\propto\epsilon^2$", fontsize=6, color="black", ha="left", va="bottom", alpha=0.5)

# draw x=1/y^2 in the second plot for a range of 1 to 1e4
x_vals = range(minimum(losscurve.x), maximum(losscurve.x), length=100)
y_vals = 1 ./ (x_vals.^2) *1.1e-5
axs[2,1].plot(x_vals, y_vals, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
# add text 1/e at the right of the top left part of line
axs[2,1].text(minimum(losscurve.x)*5, y_vals[1]/10, L"$\propto\epsilon^2$", fontsize=6, color="black", ha="left", va="bottom", alpha=0.5)

# draw x=y^2 in the third plot for a range of 1 to 1e4
x_vals = range(minimum(maxeigcurve.x), maximum(maxeigcurve.x), length=100)
y_vals = (x_vals.^2) * 6e+1
axs[2,3].plot(x_vals, y_vals, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
# add text e^2 at the right of the bottom left part of line
axs[2,3].text(maximum(maxeigcurve.x)/10, y_vals[1]*1000, L"$\propto\frac{1}{\epsilon^2}$", fontsize=6, color="black", ha="left", va="bottom", alpha=0.5)


using PyCall
for (i, ax) in fig.axes|>enumerate
    ax.set_xscale("log")
    py"""
    import matplotlib.ticker as mticker
    class MyScalarFormatter(mticker.ScalarFormatter):
        def _set_format(self):
            super()._set_format()
            self.format = "%.5g"
    """
    fmt = py"MyScalarFormatter"(useMathText=true, useOffset=true)
    fmt[:set_scientific](true)
    fmt[:set_powerlimits]((-2,2))
    fmt[:set_useLocale](true)
    ax.yaxis.set_major_formatter(fmt)
    if i == 2
        fmt.set_useOffset(-7.708)
    end
    if i == 3
        fmt.set_useOffset(0.688)
    end
    if i>3
        ax.set_xlabel(L"$1/\varepsilon$")
    end
end

fig.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.9)
fig.savefig(joinpath(proc_dir, "all_curves.pdf"))
