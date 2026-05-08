#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 6 (d) and related Appendix B
 #              eigenvalue panels. Reads the SGD trajectory results produced by
 #              `scripts/simulations/edge_of_stability_training.jl` and the
 #              shared `paper_data/edge_of_stability/initialization.dat` file
 #              (produced by the Figure 3 script — see README).
=#
using DrWatson
@quickactivate
using MLPGradientFlow, Serialization, Optimisers, Random, PyPlot, Statistics
import MLPGradientFlow: LinearSubspace, subspace_minloss, params, restart_ref, to_local_coords, cosine_similarity
using Printf
include(srcdir("helper_analyse.jl"))
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

# Load all results files directly
sims_dir = datadir("sims", "edge_of_stability")        # runtime SGD trajectories
paper_dir = projectdir("paper_data", "edge_of_stability")  # committed initialization.dat
results_files = filter(x -> startswith(x, "results-") && endswith(x, "-1.dat"), readdir(sims_dir))
edge_of_stability = [try deserialize(joinpath(sims_dir, f)) catch; nothing end for f in results_files]
edge_of_stability = reverse(sort(edge_of_stability, by = x -> x.η))
edge_of_stability = filter(res -> res !== nothing && 0.065 <= res.η <= 0.35, edge_of_stability)
min_lr = minimum(res.η for res in edge_of_stability if res !== nothing)
max_lr = maximum(res.η for res in edge_of_stability if res !== nothing)

figsize = (2.75, 2)


fig, ax = subplots(1, 1, figsize = figsize);
ax.set_ylabel("max eigval")
ax.set_xlabel("iteration")
for res in edge_of_stability
    if res == nothing; continue; end
    cmap = PyPlot.get_cmap("Oranges")
    norm_val = (log10(res.η) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    label = "η=$(Printf.@sprintf("%.1e", res.η))"
    ax.scatter(1:length(res.max_eigval), [mean(x) for x in res.max_eigval], label=label, color=color, s=1)
    ax.axhline(2/res.η, color=color, linestyle="--")
end
ax.set_yscale("log")
ax.set_xscale("log")
fig.tight_layout()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=true)
fig.savefig(joinpath(datadir("proc", "edge_of_stability"), "max_eig_small.pdf"))

# edge_of_stability for every eigenvalue, taken the first simulation only
η_list = [res.η for res in edge_of_stability if res !== nothing]
eigenval_lists = []
for run_id in 1:6
    res_list = edge_of_stability[run_id].final_samples
    eigval_list = []
    for res in res_list
        net, x = netfromres(res)
        eigval, eigvec = hessian(net, x) |> eigen
        push!(eigval_list, eigval)
    end
    eigval_list = hcat(eigval_list...)'
    push!(eigenval_lists, eigval_list)
end
cmap = PyPlot.get_cmap("Oranges")
fig, axs = subplots(size(eigval_list)[2], 6, figsize = (8, 6), sharex=true, sharey="row");
for i in 1:size(eigval_list, 2), j in 1:6
    eigval_list = eigenval_lists[j]
    ax = axs[i, j]
    if j == 1 ax.set_ylabel("eigval $i") end
    if i == 1 ax.set_title("$(Printf.@sprintf("%.2e", η_list[j]))") end
    norm_val = (log10(η_list[j]) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    ax.scatter(1:size(eigval_list, 1), eigval_list[:, i], s=2, color=color)
    if i == 4
        ax.axhline(2/η_list[j], color=color, linestyle="--")
        ax.text(1, 2/η_list[j]*1.1, "2/η", color=color, va="bottom", ha="left", fontsize=10)
    end
    ax.set_yscale("log")
    ax.set_xscale("log")
end
fig.supxlabel("iteration")
fig.suptitle("η")
fig.tight_layout()
fig.savefig(joinpath(datadir("proc", "edge_of_stability"), "all_eigvals.pdf"))
    

# vs iteration*eta
fig, ax = subplots(1, 1, figsize = figsize);
ax.set_ylabel("max eigval")
ax.set_xlabel("iteration * η")
for res in edge_of_stability
    if res == nothing; continue; end
    cmap = PyPlot.get_cmap("Oranges")
    norm_val = (log10(res.η) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    label = "η=$(Printf.@sprintf("%.1e", res.η))"
    ax.scatter((1:length(res.max_eigval))*res.η, [mean(x) for x in res.max_eigval], label=label, color=color)
    ax.axhline(2/res.η, color=color, linestyle="--")
end
ax.legend(loc="upper left", frameon=true)
ax.set_yscale("log")
ax.set_xscale("log")
fig.tight_layout()
fig.savefig(joinpath(datadir("proc", "edge_of_stability"), "max_eig2.pdf"))



# New figure: max eigval vs gamma
fig2, ax2 = subplots(1, 1, figsize = figsize);
ax2.set_xlabel("iteration")
ax2.set_ylabel("gamma")
for res in edge_of_stability
    if res == nothing; continue; end
    cmap = PyPlot.get_cmap("Oranges")
    norm_val = (log10(res.η) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    label = "η=$(Printf.@sprintf("%.1e", res.η))"
    ax2.scatter(1:length(res.max_eigval), [mean(x) for x in res.gamma], label=label, color=color)
    # ax2.axhline(2/res.η, color=color, linestyle="--")
end
ax2.legend(loc="upper left", frameon=true)
ax2.set_yscale("log")
ax2.set_xscale("log")
fig2.tight_layout()
fig2.savefig(joinpath(datadir("proc", "edge_of_stability"), "gamma.pdf"))


fig2, ax2 = subplots(1, 1, figsize = figsize);
ax2.set_xlabel("iteration * η")
ax2.set_ylabel("gamma")
for res in edge_of_stability
    if res == nothing; continue; end
    cmap = PyPlot.get_cmap("Oranges")
    norm_val = (log10(res.η) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    label = "η=$(Printf.@sprintf("%.1e", res.η))"
    ax2.scatter((1:length(res.max_eigval))*res.η, [mean(x) for x in res.gamma], label=label, color=color)
    # ax2.axhline(2/res.η, color=color, linestyle="--")
end
ax2.legend(loc="upper left", frameon=true)
ax2.set_yscale("log")
ax2.set_xscale("log")
fig2.tight_layout()
fig2.savefig(joinpath(datadir("proc", "edge_of_stability"), "gamma2.pdf"))


# compare final gamma with max_eig

fig, ax = subplots(1, 1, figsize = (3,4));
ax.set_xlabel("gamma")
ax.set_ylabel("max_eig")
# dashed gamma^2 reference line anchored at smallest gamma from highest-η result
# γ0 = mean(edge_of_stability[1].gamma[end])
# y0 = mean(edge_of_stability[1].max_eigval[end])
# γmax = mean(edge_of_stability[end].gamma[end])
γ0 = mean(edge_of_stability[end].gamma[end])
y0 = mean(edge_of_stability[end].max_eigval[end])
γmax = mean(edge_of_stability[1].gamma[end])
γline = 10 .^ range(log10(γ0), log10(γmax), length=100)
yline = y0 .* (γline ./ γ0) .^ 2
ax.plot(γline, yline, "k--", label="~γ^2", alpha=0.3, zorder=5)
yline = y0 .* (γline ./ γ0) .^ 1
ax.plot(γline, yline, "k-.", label="~γ^1", alpha=0.3, zorder=5)
for res in edge_of_stability
    if res == nothing; continue; end
    if res.η < 0.09 continue; end
    cmap = PyPlot.get_cmap("Oranges")
    norm_val = (log10(res.η) - log10(min_lr)) / (log10(max_lr) - log10(min_lr))
    norm_val = (1 - norm_val) * (0.8 - 0.2) + 0.2  # invert: scale to [0.8, 0.2] (bright to dark)
    color = cmap(norm_val)
    label = "η=$(Printf.@sprintf("%.1e", res.η))"
    ax.scatter(mean(res.gamma[end]), mean(res.max_eigval[end]), label=label, color=color)
end
ax.legend(loc="lower right", frameon=true)
ax.set_yscale("log")
ax.set_xscale("log")
fig.tight_layout()
fig.savefig(joinpath(datadir("proc", "edge_of_stability"), "gamma_v_maxeig.pdf"))


# loss landscape of γ versus max eigval
p, v2, ls = deserialize(joinpath(paper_dir, "initialization.dat"))  # load initialization (committed reference)
p += 2*v2
a = 2.017725331015221
net, x = netfromres(edge_of_stability[end-1].res[end])

# going back to gamma=1 (not on saddle because w are the ones at the channel)
x.w2[1], x.w2[2] = a, 0

gammas = collect(0:0.5:200)
alphas = collect(-0.2:0.0005:0.2)
losses = []
for i in gammas
    x.w2[1], x.w2[2] = a+i, 0-i
    eigvec, eigval = hessian(net, x) |> eigen
    max_eigvec = eigvec[:, end]
    losses_i = []
    for α in alphas
        x_temp = copy(x)
        x_temp += α*max_eigvec
        l = loss(net, x_temp)
        push!(losses_i, l)
    end
    push!(losses, losses_i)
end
losses = vcat(losses'...)

# # cap losses at cap
cap = 10
losses[losses .> cap] .= cap*1.001
# losses[log10.(losses) .> log10(cap)] .= cap


mcolors = PyPlot.matplotlib.colors
function truncated_cmap(name; minval=0.0, maxval=0.9, N=256)
    base = PyPlot.get_cmap(name)
    xs = collect(range(minval, stop=maxval, length=N))
    cols = [base(x) for x in xs]  # list of RGBA tuples
    mcolors.ListedColormap(cols)
end
trunc_viridis = truncated_cmap("viridis"; maxval=1)  # try 0.88–0.92 to taste

fig, ax = subplots(1, 1, figsize = (3.5,2), dpi=200);
ax.set_xlabel("γ")
ax.set_ylabel("α")
c = ax.contourf(gammas, alphas,losses', levels=50, cmap=trunc_viridis, vmax=cap, extend="max")
# # c = ax.pcolormesh(gammas, alphas, losses'; shading="gouraud", cmap="viridis", vmax=cap)
# c = ax.imshow(losses', extent=(minimum(gammas), maximum(gammas),
#                                minimum(alphas), maximum(alphas)),
#               origin="lower", cmap="viridis", interpolation="bilinear", aspect="auto")

cb = fig.colorbar(c, ax=ax, extend="max", label="loss")
cb.set_ticks([0, 2.5, 5, 7.5, cap])
cb.set_ticklabels(["0", "2.5", "5", "7.5", ">10"])
ax.spines["top"].set_visible(true)
ax.spines["right"].set_visible(true)

# plot trajectory of edge_of_stability[1] (does not show bouncing trajectory)
# run_no = 5
# alpha1 = []
# gamma1 = []
# last_points = edge_of_stability[run_no].final_samples[end]["trajectory"]|>values
# last_point = [params(p) for p in last_points] |> last

# net, _ = netfromres(edge_of_stability[run_no].final_samples[end])
# eigvec, eigval = hessian(net, last_point) |> eigen
# max_eigvec = eigvec[:, end]
# dxs = [p - last_point for p in params.(collect(values(edge_of_stability[run_no].final_samples[end]["trajectory"])))]
# α = [dot(max_eigvec, dx) for dx in dxs]
# gamma = [x.w2[1] / a for x in params.(collect(values(edge_of_stability[run_no].final_samples[end]["trajectory"])))]

# for i in 1:length(edge_of_stability[run_no].res)
#     net, x_sol = netfromres(edge_of_stability[run_no].res[i])
#     eigval, eigvec = hessian(net, x_sol) |> eigen
#     max_eigvec = eigvec[:, end]
#     gamma = x_sol.w2[1] / a
#     dx = x_sol - last_point
#     α = dot(max_eigvec, dx) 
#     push!(alpha1, α)
#     push!(gamma1, gamma)
# end
# ax.plot(gamma, α)

fig.tight_layout()
fig.savefig(joinpath(datadir("proc", "edge_of_stability"), "loss_landscape_shrinking_zoomout.pdf"))
