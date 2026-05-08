#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 3 (channels to infinity around a
 #              saddle line in a 4-4-1 MLP) and the trajectory data
 #              `inf9_negative.jld2` / `inf9_positive.jld2` consumed downstream.
 #
 # Workflow: pick a channel network from the scaling experiment results, merge
 # the closest pair of neurons to find a saddle direction, re-split at varying
 # μ along that direction to recover the saddle line, perturb at one μ along
 # the smallest-eigenvalue direction of the Hessian, and follow the gradient
 # flow back to the channel.
=#

using DrWatson
@quickactivate

include(srcdir("loadenv.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("helper_retrievers.jl"))

exp_name = "scaling"
df_setup, sims_dir, proc_dir = setup_results(exp_name; sims_dir = projectdir("paper_data", exp_name))

layer_dists = [generate_retrieve_closest_pair(w, m) for w in ["w1", "w2", "w3", "w4"] for m in [cos_dist, l2_dist]]
retrievers = [retrieve_gnorm, retrieve_weight_norms, retrieve_x, retrieve_loss, layer_dists...]

df = retrieve_infos(df_setup, retrievers=retrievers, verbose=true)

# filter out f=softplus and wnorm that are below 1e3
df = filter(row -> row[:f] != softplus && row[:wnorm] > 1e3 && row[:r] == 4 && row[:Din] == 4, df)

# Pick the network used in the published Figure 3. The committed
# `paper_data/scaling/` directory contains exactly that one file, so df has a
# single row after filtering. (Originally this was `df[9, :]` of the full
# scaling sweep — see README for how to regenerate the source file.)
row = df[1, :]
res = load(row.path)["res"]
net, _ = netfromres(res)

# write merge_neurons function
function merge_neurons(w, a, pair)
    # merging two neurons means averaging their input weights (w) and summing their output weights (a)
    w = deepcopy(w)
    a = deepcopy(a)
    i, j = pair
    w[i, :] = (w[i, :] + w[j, :]) / 2
    a[:, i] = a[:, i] + a[:, j]
    w = w[setdiff(1:size(w,1), j), :] # remove j-th row of w 
    a = a[:, setdiff(1:size(a,2), j)] # remove j-th column of a
    w, a
end

# merge pair of neurons with smallest distance
w1 = row.x["w1"]
w2 = row.x["w2"]
f = row.f
# remove neuron pair give by pair_cos_dist_w1 and replace with their average
w, a = merge_neurons(w1, w2, row.pair_cos_dist_w1)
x = Dict("w1" => w, "w2" => a) |> params
merged_spec = ((size(x.w1)[1], f, true),(1, identity, true))
merged_net = Net(layers = merged_spec,
                 bias_adapt_input = false,
                 input = res["input"],
                 target = res["target"],
                 derivs = 2)
loss(merged_net, x)

exp_name = "fig3"
mkpath(datadir("sims", exp_name))
inf9_path = datadir("sims", exp_name, "inf9.jld2")

# Train to converge to closest local min, save, then load. Skip the train+save if cache exists.
if !isfile(inf9_path)
    new_res = train(merged_net, x,
                    maxtime_ode = 180,
                    maxtime_optim = 180,
                    patience = 10^6,
                    maxiterations_ode = 10^5,
                    maxiterations_optim = 10^5,
                    maxnorm = 10^3,
                    progress_interval = 5,
                    n_samples_trajectory = 5,
                    g_tol = 0.0,
                    reltol = 1e-3,
                    minloss = 1e-30,
                    verbosity = 1)
    tagsave(inf9_path,
            Dict("res" => new_res,
                 "setting" => row,
                 "unmerged_x" => row.x,
                 "unmerged_path" => row.path))
end

new_res = load(inf9_path)["res"]
new_net, _ = netfromres(new_res)

a_t = new_res["x"]["w2"][1]

# μ_approx = (abs(w2[1])+abs(w2[2]))/2a_t  #TODO: this needs to be fixed
μ_approx = (w2[1]-w2[2]+a_t)/2a_t
μ_approx = (w2[1]+a_t-w2[2])/2a_t 
μ_approx = 1.5

saddle_x = split(new_res["x"]|>params, 1, μ_approx) |>params2dict

# permute the weights to match the original network
saddle_x["w1"][2, :], saddle_x["w1"][4, :] = saddle_x["w1"][4, :], saddle_x["w1"][2, :]
# saddle_x["w2"][:, 4], saddle_x["w2"][:, 5] = saddle_x["w2"][:, 5], saddle_x["w2"][:, 4]
saddle_x["w2"][:, 2], saddle_x["w2"][:, 4] = saddle_x["w2"][:, 4], saddle_x["w2"][:, 2]

# compute loss over interpolation between saddle_x and row.x

xs = params(saddle_x)
xi = params(row.x)

loss(net, xs), loss(net, xi)

# n_steps = 100
# losses = []
# alphas = []
# for n in -n_steps:1:3n_steps
#     α = n/2n_steps
#     xn = (1-α)*xs + α*xi
#     println(loss(net, xn))
#     push!(losses, loss(net, xn))
#     push!(alphas, α)
# end
# plot(alphas, losses)

exp_name = "fig3"
sims_dir, proc_dir = mkpath(datadir("sims", exp_name)), mkpath(datadir("proc", exp_name))

# compute evecs at μ_approx
evals, evecs = hessian(net, xs)|>eigen
# compute projection of xi on the negative eigenvalue direction
dot(evecs[:, 1], xi)

gradient(net, xs)|>norm

p = evecs[:, 1]*0.1

loss(net, xs), loss(net, xs+p)

# Compute the two perturbation trajectories from the saddle along ±p (the smallest-eigenvalue direction)
# and cache them. The downstream plotting code loads `resp` (negative direction) and `resp2` (positive).
inf9_neg_path = datadir("sims", exp_name, "inf9_negative.jld2")
inf9_pos_path = datadir("sims", exp_name, "inf9_positive.jld2")

if !isfile(inf9_neg_path)
    resp_train = train(net, xs-p,
                       maxtime_ode = 60,
                       maxtime_optim = 60,
                       patience = 10^6,
                       maxiterations_ode = 10^5,
                       maxiterations_optim = 10^5,
                       maxnorm = 10^3,
                       progress_interval = 5,
                       n_samples_trajectory = 5,
                       g_tol = 0.0,
                       reltol = 1e-3,
                       minloss = 1e-30,
                       verbosity = 1)
    tagsave(inf9_neg_path, resp_train)
end

if !isfile(inf9_pos_path)
    resp_train = train(net, xs+p,
                       maxtime_ode = 60,
                       maxtime_optim = 60,
                       patience = 10^6,
                       maxiterations_ode = 10^5,
                       maxiterations_optim = 10^5,
                       maxnorm = 10^3,
                       progress_interval = 5,
                       n_samples_trajectory = 5,
                       g_tol = 0.0,
                       reltol = 1e-3,
                       minloss = 1e-30,
                       verbosity = 1)
    tagsave(inf9_pos_path, resp_train)
end

resp  = load(inf9_neg_path)
resp2 = load(inf9_pos_path)

######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
using PyPlot

# make a lineplot using  the only negative eigenvalue direction from the saddle.
fig, ax = subplots(1,1,figsize=(3, 1.5))
cmap = get_cmap("viridis_r")
# choose μs log-spaced out from 1 to 25
μs = exp10.(range(log10(1), log10(3), length=20))

for (i, μ) in enumerate(μs)
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]

    evals, evecs = hessian(net, x_plat)|>eigen
    # plot the loss along the negative eigenvalue direction
    n_steps = 100
    losses = []
    alphas = []
    for n in -n_steps:1:n_steps
        α = n/2n_steps
        xn = x_plat + evecs[:, 1]*α
        push!(losses, loss(net, xn))
        push!(alphas, α)
    end
    ax.plot(alphas, losses, color=cmap(i/length(μs)))
    ax.set_ylim([minimum(losses)*0.995, loss(net, xs)*1.01])
    ax.set_xlim([minimum(alphas), maximum(alphas)])
end
# add legend as colorbar
sm = PyPlot.ScalarMappable(cmap=cmap, norm=PyPlot.colorsm.Normalize(vmin=minimum(μs), vmax=maximum(μs)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("μ", rotation=0)
xlabel("α")
ylabel("loss")

######################## set scientific notation for xticks########################
using PyCall
py"""
import matplotlib.ticker as mticker
class MyScalarFormatter(mticker.ScalarFormatter):
    def _set_format(self):
        super()._set_format()
        # Enforce two-significant-digits format
        self.format = "%.3g"
"""
fmt = py"MyScalarFormatter"(useMathText=true)
fmt[:set_scientific](true)
fmt[:set_powerlimits]((-2,2))
ax.yaxis.set_major_formatter(fmt)
########################################################################

fig.tight_layout(pad=0)
# fig.savefig(datadir(proc_dir, "loss_along_eigen.pdf"))

###########################################################################

# actual 3d surface plot
# μs_left = range(-2.5, -0.5, step=0.1) 
μs_mid = range(1, 1.5, step=0.025)
μs_right = range(1.5, 3, step=0.1) 
μs = unique([μs_mid..., μs_right...])

# μs = 0.8:0.1:3
α_min, α_max = -0.25, 0.25 

losses = []
alphas_all = []
for (i, μ) in enumerate(μs)
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]

    evals, evecs = hessian(net, x_plat)|>eigen

    α_res = 1001                  
    αs = range(α_min, α_max, length=α_res)
    losses_x = []
    for α in αs
        xn = x_plat + evecs[:, 1]*α
        push!(losses_x, loss(net, xn))
    end
    push!(losses, losses_x)
    push!(alphas_all, αs)
    print(".")
end
losses = hcat(losses...)
alphas_all = hcat(alphas_all...)

X = repeat(μs', size(alphas_all, 1), 1)
Y = alphas_all
Z = losses

# Define the semi-major and semi-minor axes of the ellipse
a = 3.5  # Semi-major axis
b = 0.225  # Semi-minor axis
# Create an elliptical mask
center_x, center_y = 1, 0
p = 3
mask = abs.((X .- center_x) ./ a).^p .+ abs.((Y .- center_y) ./ b).^p .<= 1
# Apply the mask to the data
Z_masked = copy(Z)
# Z_masked[.!mask] .= NaN  # Set values outside the mask to NaN
Z_masked[Z_masked .> 0.057] .= NaN  # Set negative values to NaN
# # put to NaN values of Z_masked that are above 0.0544 AND placed where α is < -0.12 AND μ is > 1.1
# Z_masked[(Z_masked .> 0.0548) .& (Y .< -0.14) .& (X .> 1.1)] .= NaN


fig = figure(figsize=(2.5,2.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z_masked, cmap="viridis", edgecolor="none", linewidth=0.0, antialiased=true)

ax.set_ylabel("α")
ax.set_xlabel("μ")
ax.set_zlabel("loss")
ax.grid(false)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks([1, 1.5, 2, 2.5, 3])
ax.set_xlim([1, 3])
ax.set_ylim([α_min, α_max])
ax.set_yticks([-0.20, 0, 0.20])
# make sure the zticks are only 3
ax.set_zticks([minimum(Z), Z[101,1], 0.057])

######################## set scientific notation for zticks########################
using PyCall
py"""
import matplotlib.ticker as mticker
class MyScalarFormatter(mticker.ScalarFormatter):
    def _set_format(self):
        super()._set_format()
        # Enforce two-significant-digits format
        self.format = "%.2g"
"""
fmt = py"MyScalarFormatter"(useMathText=true)
fmt[:set_scientific](true)
fmt[:set_powerlimits]((-2,2))
ax.zaxis.set_major_formatter(fmt)
########################################################################
ax.set_zlim([minimum(Z), 0.058])

ax.view_init(35, -40)
ax.set_xticks([1, 2, 3])
ax.tick_params(axis="x", pad=-2) 
ax.set_yticks([-0.2, 0, 0.2])
ax.tick_params(axis="y", pad=-2) 
ax.set_zticks([0.055, 0.057])
ax.tick_params(axis="z", pad=-2)
fig.set_size_inches(2.5, 2.5)
fig.tight_layout(pad=0)
fig.set_size_inches(2.5, 2.5)
# fig.savefig(datadir(proc_dir, "loss_surface2.pdf"))



########################################################################


###########################################################################

# longer 3d surface plot
μs_left = range(-2.5, -0.5, step=0.1) 
μs_mid = range(-0.5, 1.5, step=0.001)
μs_right = range(1.5, 3, step=0.1) 
μs = unique([μs_left..., μs_mid..., μs_right...])

# μs = 0.8:0.1:3
α_min, α_max = -0.25, 0.25 

losses = []
alphas_all = []
for (i, μ) in enumerate(μs)
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]

    evals, evecs = hessian(net, x_plat)|>eigen

    α_res = 1001                  
    αs = range(α_min, α_max, length=α_res)
    losses_x = []
    for α in αs
        xn = x_plat + evecs[:, 1]*α
        push!(losses_x, loss(net, xn))
    end
    push!(losses, losses_x)
    push!(alphas_all, αs)
    print(".")
end
losses = hcat(losses...)
alphas_all = hcat(alphas_all...)

X = repeat(μs', size(alphas_all, 1), 1)
Y = alphas_all
Z = losses

# Define the semi-major and semi-minor axes of the ellipse
a = 3.5  # Semi-major axis
b = 0.225  # Semi-minor axis
# Create an elliptical mask
center_x, center_y = 1, 0
p = 3
mask = abs.((X .- center_x) ./ a).^p .+ abs.((Y .- center_y) ./ b).^p .<= 1
# Apply the mask to the data
Z_masked = copy(Z)
# Z_masked[.!mask] .= NaN  # Set values outside the mask to NaN
Z_masked[Z_masked .> 0.057] .= NaN  # Set negative values to NaN
# # put to NaN values of Z_masked that are above 0.0544 AND placed where α is < -0.12 AND μ is > 1.1
# Z_masked[(Z_masked .> 0.0548) .& (Y .< -0.14) .& (X .> 1.1)] .= NaN


fig = figure(figsize=(5.5,5.5), dpi=250)
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect((2, 1, 1.25))  # (x, y, z) aspect ratio; stretch x relative to y and z
ax.plot_surface(X, Y, Z_masked, cmap="viridis", edgecolor="none", linewidth=0.0, antialiased=true)

ax.set_ylabel("α")
ax.set_xlabel("μ")
ax.set_zlabel("loss")
ax.grid(false)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks([-2, 0, 1, 3])
ax.set_xlim([-2, 3])
ax.set_ylim([α_min, α_max])
ax.set_yticks([-0.20, 0, 0.20])
# make sure the zticks are only 3
ax.set_zticks([minimum(Z), Z[101,1], 0.057])

######################## set scientific notation for zticks########################
using PyCall
py"""
import matplotlib.ticker as mticker
class MyScalarFormatter(mticker.ScalarFormatter):
    def _set_format(self):
        super()._set_format()
        # Enforce two-significant-digits format
        self.format = "%.2g"
"""
fmt = py"MyScalarFormatter"(useMathText=true)
fmt[:set_scientific](true)
fmt[:set_powerlimits]((-2,2))
ax.zaxis.set_major_formatter(fmt)
########################################################################
ax.set_zlim([minimum(Z), 0.058])

ax.view_init(40, -20)
ax.tick_params(axis="x", pad=2) 
ax.set_yticks([-0.2, 0, 0.2])
ax.tick_params(axis="y", pad=2) 
ax.set_zticks([0.055, 0.057])
ax.tick_params(axis="z", pad=2)
fig.set_size_inches(5.5, 3.5)
fig.tight_layout(pad=0)
# fig.savefig(datadir(proc_dir, "loss_surface_stretched.pdf"))



########################################################################



μs = 10 .^ (0:0.01:5)
α_min, α_max = -0.25, 0.25 

losses = []
alphas_all = []
for (i, μ) in enumerate(μs)
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]

    evals, evecs = hessian(net, x_plat)|>eigen

    α_res = 201                  
    αs = range(α_min, α_max, length=α_res)
    losses_x = []
    for α in αs
        xn = x_plat + evecs[:, 1]*α
        push!(losses_x, loss(net, xn))
    end
    push!(losses, losses_x)
    push!(alphas_all, αs)
end
losses = hcat(losses...)
alphas_all = hcat(alphas_all...)

X = repeat(μs', size(alphas_all, 1), 1)
Y = alphas_all
Z = losses
########################################################################

# make 2d plot of the surface
fig, ax = subplots(1,1,figsize=(3,1))
cmap = get_cmap("viridis")

ax.contourf(X', Y', Z', cmap=cmap, levels=range(minimum(Z), 0.0555, length=100), extend="both")

ax.set_ylabel("α")
ax.set_xlabel("μ")
ax.set_yticks([-0.2, 0, 0.2])
ax.set_xscale("log")
# ax.set_yticks([1, 1.5, 2, 2.5, 3])

fig.tight_layout(pad=0)
fig.set_size_inches(3, 1)
# fig.savefig(datadir(proc_dir, "loss_surface2D.pdf"))
########################################################################


norm(params(resp["x"]) - xs)
norm(params(resp["x"]) - xi)

cos_sim(x, y) = dot(x, y)/(norm(x)*norm(y))

# (resp, resp2 — the ±p perturbation trajectories — are computed/loaded near the top of this script.)

# compute approximation of μ for any solution in "trajectory"
μs = []
for (i, (_, W)) in enumerate(resp["trajectory"])
    a_1 = W["w2"][1]
    a_2 = W["w2"][2]
    μ_approx = (a_1-a_2+a_t)/2a_t
    push!(μs, μ_approx)
    println(μ_approx)
end
# compute the same but for resp2
μs2 = []
for (i, (_, W)) in enumerate(resp2["trajectory"])
    a_1 = W["w2"][1]
    a_2 = W["w2"][2]
    μ_approx = (a_1-a_2+a_t)/2a_t
    push!(μs2, μ_approx)
    println(μ_approx)
end

# plot loss for each point of trajectory
fig, ax = subplots(1,1,figsize=(2.5,1))
ax.plot(μs, resp["loss_curve"], color="green", label=L"\alpha_{-}")
ax.scatter(μs[end], resp["loss_curve"][end], c="green", marker=9)
ax.plot(μs2, resp2["loss_curve"][1:end-1], color="orange", label=L"\alpha_{+}")
ax.scatter(μs2[end], resp2["loss_curve"][end-1], c="orange", marker="x")
ax.set_xscale("log")
ax.set_xlabel("μ")
ax.set_ylabel("loss")
ax.set_xticks([1, 1.5, 3, 5, 10, 25])
ax.set_xticklabels(["1", "1.5", "3", "5", "10", "25"])
ax.legend(loc="upper right", borderpad=0.2, borderaxespad=0.2,handletextpad=0.2)
fig.tight_layout(pad=0)
fig.set_size_inches(2.5, 1)
# fig.savefig(datadir(proc_dir, "loss_along_trajectory.pdf"))

fig, ax = subplots(1,1,figsize=(2.5,1))
ax.plot(μs, resp["loss_curve"].-resp["loss"], color="green", label=L"e_{-}")
ax.scatter(μs[end], resp["loss_curve"][end]-resp["loss"], c="green", marker=9)
ax.plot(μs2, resp2["loss_curve"][1:end-1].-resp["loss"], color="orange", label=L"e_{+}")
ax.scatter(μs2[end], resp2["loss_curve"][end-1]-resp["loss"], c="orange", marker="x")
ax.set_xscale("log")
ax.set_xlabel("μ")
ax.set_ylabel(L"loss - loss$_\infty$")
ax.set_xticks([1, 1.5, 3, 5, 10, 25])
ax.set_xticklabels(["1", "1.5", "3", "5", "10", "25"])
ax.set_yscale("log")
ax.set_yticks([1e-1, 1e-4, 1e-7])
fig.tight_layout(pad=0)
fig.set_size_inches(2.5, 1)
# fig.savefig(datadir(proc_dir, "loss_along_trajectory_log.pdf"))

# plot gradient norm for each point of the trajectory 
# TODO: to make it work locally I need to downgrade the famous VectorizationBase package...
fig, ax = subplots(1,1,figsize=(4,2))
ax.plot(μs, [gradient(net, params(W))|>norm for (_, W) in resp["trajectory"]], color="green", label=L"e_{-}")
ax.scatter(μs[end], gradient(net, params(resp["x"]))|>norm, c="green", marker=9)
ax.plot(μs2, [gradient(net, params(W))|>norm for (_, W) in resp2["trajectory"]], color="orange", label=L"e_{+}")
ax.scatter(μs2[end], gradient(net, params(resp2["x"]))|>norm, c="orange", marker="x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("μ")
ax.set_ylabel(L"$||\nabla \mathcal{L}||$")
ax.set_xticks([1, 1.5, 3, 5, 10, 25])
ax.set_xticklabels(["1", "1.5", "3", "5", "10", "25"])
fig.tight_layout()

# compute parameter norm distance between samples of trajectory and split at corresponding μ
distances = []
for (i, (_, W)) in enumerate(resp["trajectory"])
    μ = μs[i]
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]
    push!(distances, norm(params(x_plat) - params(W)))
end
distances2 = []
for (i, (_, W)) in enumerate(resp2["trajectory"])
    μ = μs2[i]
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]
    push!(distances2, norm(params(x_plat) - params(W)))
end

fig, ax = subplots(1,1,figsize=(4,2))
ax.plot(μs, distances, color="green", label=L"e_{-}")
ax.scatter(μs[end], distances[end], c="green", marker=9)
ax.plot(μs2, distances2, color="orange", label=L"e_{+}")
ax.scatter(μs2[end], distances2[end], c="orange", marker="x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("μ")
ax.set_ylabel(L"$||x - x_{plat}||$")
ax.set_xticks([1, 1.5, 3, 5, 10, 25])
ax.set_xticklabels(["1", "1.5", "3", "5", "10", "25"])
fig.tight_layout()
# fig.savefig(datadir(proc_dir, "distance_along_trajectory.pdf"))

# track cosine distance and norm of individual neurons in plateau and trajectory
w_norms = []
w_cosines = []
for (i, (_, W)) in enumerate(resp["trajectory"])
    μ = μs[i]
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]
    norms = []
    cosines = []
    for (wt, wp) in zip(W["w1"]|>eachrow, x_plat.w1|>eachrow)
        push!(norms, norm(wt - wp))
        push!(cosines, cos_dist(wt, wp))
    end
    push!(w_norms, norms)
    push!(w_cosines, cosines)
end
w_norms = hcat(w_norms...)
w_cosines = hcat(w_cosines...)

# track cosine distance and norm of individual neurons in plateau and trajectory for resp2
w_norms2 = []
w_cosines2 = []
for (i, (_, W)) in enumerate(resp2["trajectory"])
    μ = μs2[i]
    x_plat = split(new_res["x"]|>params, 1, μ)
    x_plat.w1[2, :], x_plat.w1[4, :] = x_plat.w1[4, :], x_plat.w1[2, :]
    # x_plat.w2[:, 4], x_plat.w2[:, 5] = x_plat.w2[:, 5], x_plat.w2[:, 4]
    x_plat.w2[:, 2], x_plat.w2[:, 4] = x_plat.w2[:, 4], x_plat.w2[:, 2]
    norms = []
    cosines = []
    for (wt, wp) in zip(W["w1"]|>eachrow, x_plat.w1|>eachrow)
        push!(norms, norm(wt - wp))
        push!(cosines, cos_dist(wt, wp))
    end
    push!(w_norms2, norms)
    push!(w_cosines2, cosines)
end
w_norms2 = hcat(w_norms2...)
w_cosines2 = hcat(w_cosines2...)


fig, ax = subplots(1,1,figsize=(4,2.5), dpi=200)
ax2 = ax.twinx()
# Define line styles for different neurons
line_styles = ["-", "-", "--", ":"]
# Plot w_norms with different line styles
for i in 1:size(w_norms,1)
    col1 = i<3 ? (0.3, 0.4, 0, 1) : (0.3, 0.4, 0, 0.5)  # Warm green (more red)
    col2 = i<3 ? (0.2, 0.6, 0.2, 1) : (0.2, 0.6, 0.2, 0.5)  # Cool green (more blue)
    ax.plot(μs, w_norms[i,:], color=col1, linestyle=line_styles[i], label=L"w_{%$i}")
    ax2.plot(μs, w_cosines[i,:], color=col2, linestyle=line_styles[i])
end
# Set scales and labels
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("μ")
ax.set_ylabel(L"$||w^{i} - w^{i}_{\text{plat}}||$", color=(0.3, 0.4, 0, 1))
ax.set_ylim([5e-2, 2.75])
ax2.set_ylabel("Cosine Similarity", color=(0.2, 0.6, 0.2, 1))
ax2.set_ylim([0, 0.17])
ax.set_xlim([minimum(μs), maximum(μs)])
ax2.set_xlim([minimum(μs), maximum(μs)])
ax.tick_params(axis="y")
ax2.tick_params(axis="y")
# Set x-ticks and labels
ax.set_xticks([1.5, 3, 5, 10, 25])
ax.set_xticklabels(["1.5", "3", "5", "10", "25"])
ax.spines["bottom"].set_bounds(minimum(μs), maximum(μs))
ax2.spines["bottom"].set_bounds(minimum(μs), maximum(μs))
ax.spines["bottom"].set_position(("outward", 3))
ax2.spines["bottom"].set_position(("outward", 3))
ax.spines["left"].set_position(("outward", 3))
ax2.spines["right"].set_position(("outward", 3))
# remove up and side spine
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
ax2.spines["top"].set_visible(false)
ax2.spines["left"].set_visible(false)

# Add legends
legend1 = ax.legend(loc="upper right", framealpha=1)
for (i, (line, txt)) in zip(legend1.get_lines(), legend1.get_texts()) |> enumerate
    if i < 3 line.set_color("black"), txt.set_color("black")
    else line.set_color((0,0,0,0.5)), txt.set_color((0,0,0,0.5)) end
end
fig.tight_layout()
# fig.savefig(datadir(proc_dir, "distance_neurons_along_trajectory.pdf"))

############################################
############################################
############################################
############################################
############################################

fig, ax = subplots(1,1,figsize=(4,2.5), dpi=200)
ax2 = ax.twinx()
# Define line styles for different neurons
line_styles = ["-", "-.", "--", ":"]
# Plot w_norms with different line styles
for i in 1:size(w_norms,1)
    col3 = (0.7, 0.35, 0.0, 1) 
    col4 = "orange"
    ax.plot(μs2, w_norms2[i,:], color=col3, linestyle=line_styles[i])
    ax2.plot(μs2, w_cosines2[i,:], color=col4, linestyle=line_styles[i], label=L"w_{%$i}")
end
# Set scales and labels
ax.set_yscale("log")
ax.set_xlabel("μ")
ax.set_ylabel(L"$||w^{i} - w^{i}_{\text{plat}}||$", color=(0.7, 0.35, 0.0, 1) )
ax.set_ylim([5e-5, 50])
ax2.set_ylabel("Cosine Similarity", color="orange")
ax2.set_ylim([0, 0.1])
ax.set_xlim([0.9, maximum(μs2)])
ax2.set_xlim([0.9, maximum(μs2)])
ax.tick_params(axis="y")
ax2.tick_params(axis="y")
# Set x-ticks and labels
ax.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
ax.spines["bottom"].set_bounds(0.9, maximum(μs2))
ax2.spines["bottom"].set_bounds(0.9, maximum(μs2))
ax.spines["bottom"].set_position(("outward", 3))
ax2.spines["bottom"].set_position(("outward", 3))
ax.spines["left"].set_position(("outward", 3))
ax2.spines["right"].set_position(("outward", 3))
# remove up and side spine
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
ax2.spines["top"].set_visible(false)
ax2.spines["left"].set_visible(false)

# Add legends
legend1 = ax2.legend(loc="upper left", framealpha=1)
for (i, (line, txt)) in zip(legend1.get_lines(), legend1.get_texts()) |> enumerate
    line.set_color("black"), txt.set_color("black")
end
fig.tight_layout()
# fig.savefig(datadir(proc_dir, "distance_neurons_along_trajectory2.pdf"))


###############################################

# plot cosine distance of gradient and μ line
p1 = split(new_res["x"]|>params, 1, 1.5)
# p1.w1[2, :], p1.w1[4, :] = p1.w1[4, :], p1.w1[2, :]
# p1.w2[:, 4], p1.w2[:, 5] = p1.w2[:, 5], p1.w2[:, 4]
# p1.w2[:, 2], p1.w2[:, 4] = p1.w2[:, 4], p1.w2[:, 2]

p2 = split(new_res["x"]|>params, 1, 2.5)
# p2.w1[2, :], p2.w1[4, :] = p2.w1[4, :], p2.w1[2, :]
# p2.w2[:, 4], p2.w2[:, 5] = p2.w2[:, 5], p2.w2[:, 4]
# p2.w2[:, 2], p2.w2[:, 4] = p2.w2[:, 4], p2.w2[:, 2]

μ_line = p2 - p1
trj = resp["trajectory"]|>Tuple;

Δupdates = [cos_sim(params(trj[i][2]) - params(trj[i-1][2]), μ_line) for i in 2:length(trj)-1]
    

fig, ax = subplots(1,1,figsize=(2.5,1), dpi=200)
ax.plot(collect(keys(resp["trajectory"]))[1:end-2], Δupdates, color="green")
ax.set_xscale("log")
ax.set_xlabel(L"$t_{\text{ODE}}$")
ax.set_ylabel(L"$\cos(\Delta x, \Gamma)$")
ax.axhline(1, color="grey", linestyle="--",zorder=0)
fig.tight_layout(pad=0)
fig.savefig(datadir(proc_dir, "cosine_gradient_mu.pdf"))

# same figure but with μ on the x axis
μs = []
for (i, (_, W)) in enumerate(resp["trajectory"])
    a_1 = W["w2"][1]
    a_2 = W["w2"][2]
    μ_approx = (a_1-a_2+a_t)/2a_t
    push!(μs, μ_approx)
end

fig, ax = subplots(1,1,figsize=(2.5,1), dpi=200)
ax.plot(μs[1:end-2], Δupdates, color="green")
ax.set_xscale("log")
ax.set_xlabel(L"$\mu$")
ax.set_ylabel(L"$\cos(\Delta x, \Gamma)$")
ax.axhline(1, color="grey", linestyle="--",zorder=0)
ax.set_xlim([0.8, 25])
ax.set_xticks([1, 10])
ax.set_xticklabels(["1", "10"])
fig.tight_layout(pad=0)
fig.savefig(datadir(proc_dir, "cosine_gradient_mu2.pdf"))