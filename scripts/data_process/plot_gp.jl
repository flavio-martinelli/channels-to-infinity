#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the GP-teacher illustration plots used in Appendix B
 #              (a 2D scatter of GP teacher 2 samples, and a quadratic surface).
=#
using DrWatson
@quickactivate

include(srcdir("funs_teachers.jl"))
include(srcdir("helper_experiment.jl")) # for functionning of gp_teacher
include(srcdir("colors.jl"))
using PyPlot, MLPGradientFlow, StatsBase
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

proc_dir = mkpath(datadir("proc", "plot_gp"))

x, y, _ = gp_teacher2(Nsamples=3000, Din=2, seed=2991)
# scatter plot of x points in y color
fig, ax = subplots(1,1,figsize=(3,3))
sc = ax.scatter(x[1, :], x[2, :], c=y, cmap="viridis", s=20)
# turn off all axes and spines
ax.set_axis_off()
fig.tight_layout()
fig.savefig(joinpath(proc_dir, "gp_teacher2_scatter.pdf"))

# 3dplot surface of 3d paraboloid
xx = range(-3, 3, length=100)
yy = range(-3, 3, length=100)
XX = repeat(collect(xx)', 100, 1)
YY = repeat(collect(yy), 1, 100)
Z = XX.^2 + YY.^2
fig = figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(XX, YY, Z, cmap="viridis", edgecolor="none")
fig.savefig(joinpath(proc_dir, "quadratic_surface.pdf"))
