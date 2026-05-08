#=
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Produces the plots for Figure 2 (c, d) (neuron-splitting saddle
 #              perturbation analysis) and the related Appendix A.2 saddle-line
 #              figure. Reads pre-computed perturbation simulations from
 #              `data/sims/fig1A_saddle_perturbation/<ID>/split_<split_id>/`.
=#

using DrWatson
@quickactivate

using MLPGradientFlow, DataFrames, PyPlot, Serialization, OrderedCollections

include(srcdir("loadenv.jl"))
include(srcdir("helper_analyse.jl"))
include(srcdir("funs_teachers.jl"))
include(srcdir("helper_plot.jl"))


######################################################
######################################################
######################################################
######################################################
#################      CONFIGS        ################
######################################################
######################################################
######################################################
######################################################

PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

ID = 86  # fig1A example id
split_id = 2  # index of the splitted neuron

exp_name = joinpath("fig1A_saddle_perturbation","$(ID)", "split_$(split_id)")
sim_dir = projectdir("paper_data", exp_name)  # committed reference perturbation results
proc_dir = datadir("proc", exp_name)
mkpath(proc_dir)


######################################################
######################################################
######################################################
######################################################
############      DATA EXTRACTION        #############
######################################################
######################################################
######################################################
######################################################

df = DataFrame( 
    μ = Float64[],  # splitting parameter 
    ℒ = Float64[],  # loss
    x = Any[],  # parameters before splitting
    init = Any[],  # parameters after splitting and after perturbation
    xs = Any[],  # parameters after splitting before perturbation
    trj = Any[],  # learning trajectory: parameters
    loss_trj = Any[],  # learning trajectory: loss
    distances = Any[],  # learning trajectory: L2 distance to μline
    μproj = Any[],  # learning trajectory: projection on μline
)

## load res of simulated perturbations
println("loading splitting simulation data")
for (i, d) in enumerate(readdir(sim_dir))
    data = deserialize(joinpath(sim_dir, d))
    res = data.res
    μ = data.μ
    perturbation_strength = data.perturbation_strength[1]
    trjEuler = data.trj|>values|>collect
    trjODE = res["trajectory"]|>values|>collect
    trj = [trjEuler..., trjODE...]
    loss_trj = data.loss_trj
    ℒ = res["loss"]
    x = res["x"]
    xs = data.xs
    init = res["init"]
    k = (x["w1"]|>size)[1]
    push!(df, (μ, ℒ, x, init, xs, trj, loss_trj, [0.0], [0.0]))
    if i%10 == 0 print(".") end
    # println(res["optim_stopped_by"])
    # println("converged: $(res["converged"])")
    # TODO: appendix script that analyses the goodness of training (convergence, gradients, etc.)
end

## sort by μ
sort!(df, :μ)

## number of neurons k after split 
k = df[1, :x]["w1"]|>size|>first

## retrieve general info from first file
data = deserialize(joinpath(sim_dir, readdir(sim_dir)[1]))
perturbation_strength = data.perturbation_strength
xs = data.xs|>params2dict
x_before_split = data.xs|>params2dict
x_before_split["w1"] = x_before_split["w1"][1:k-1, :]
x_before_split["w2"] = x_before_split["w2"][:, 1:k-1]
x_before_split["w2"][1, split_id] += xs["w2"][1, end]
# TODO: new simulations will save both x and xs, no need to extract it

xx = data.res["input"]
yy = data.res["target"]

######################################################
######################################################
######################################################
######################################################
############      PLOT OF SOLUTIONS      #############
######################################################
######################################################
######################################################
######################################################

println("plotting solutions")

# A: space left for graphics of saddle split
# B: unsplit solution, C: arrows and text
# D: split solution μ ∈ [0, 1] E: split solution μ ∉ [0, 1]
# F: loss plot as function of μ
# G: trajectory in parameter space after perturbation
# H: loss surface plot
# I: eigenvalues plot as function of μ
# L: rotation of eigenvalues plot as funciton of μ

mosaic = [["A", "A", "A", "A", "A", "F", "F", "F", "F", "H", "H", "H"],
          ["A", "A", "A", "A", "A", "F", "F", "F", "F", "H", "H", "H"],  
          ["B", "B", "C", "D", "D", "F", "F", "F", "F", "I", "I", "I"],
          ["B", "B", "C", "D", "D", "G", "G", "G", "G", "I", "I", "I"],
          ["B", "B", "C", "E", "E", "G", "G", "G", "G", "L", "L", "L"],
          ["B", "B", "C", "E", "E", "G", "G", "G", "G", "L", "L", "L"]]

fig, axs = plt.subplot_mosaic(mosaic, dpi=150, layout="constrained") 
fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0.0, wspace=0.0)
fig.set_size_inches(6, 3);  
# fig.set_layout_engine("none")

########### A: space left for graphics of saddle split ###########

ax = axs["A"]
ax.set_position([0.0, 2/3, 2.5/6, 1/3])
# keep spines but remove ticks
for spine in ["top", "right", "left", "bottom"]
    ax.spines[spine].set_visible(true)
end
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# add text as title of right-top plot
ax.text(4/5, 0.1, "After split", ha="center", va="center", transform=ax.transAxes)

########### B: unsplit solution ###########

ax = axs["B"]

x_margin = 0.01
y_margin = 0.01

ax.axis("off")

# generate inset at x=0, y=0.25
ax_inset = ax.inset_axes([x_margin, 0.25+y_margin, 1-2*x_margin, 0.5-2*y_margin/2])  # x, y, width, height

plot_network(x_before_split|>params, g; ax=ax_inset, clims=[-3.5, 2.1],
                                        bias=false, put_colorbar=false)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
for spine in ["top", "right", "left", "bottom"]
    ax_inset.spines[spine].set_visible(true)
end
ax.text(0.5, 0.8, "Original", ha="center", transform=ax.transAxes)

########### C: arrows and text ###########

ax = axs["C"]
ax.set_position([1/6, 0, 0.5/6, 2/3])
ax.axis("off")

# draw an arrow from mid left to upper right
ax.annotate("", xy=(0.95, 0.60), xytext=(0.05, 0.60),
            arrowprops=Dict("width" => 0.005, "color" => "black",
                            "headwidth" => 2.5, "headlength" => 4))
# add text above the arrow
ax.text(0.5, 0.63, L"\mu \in (0, 1)", ha="center", va="bottom", transform=ax.transAxes, fontsize=7)

ax.annotate("", xy=(0.95, 0.40), xytext=(0.05, 0.40),
            arrowprops=Dict("width" => 0.005, "color" => "black",
                            "headwidth" => 2.5, "headlength" => 4))

ax.text(0.5, 0.37, L"\mu \notin (0, 1)", ha="center", va="top", transform=ax.transAxes, fontsize=7)

########### D: split solution μ ∈ [0, 1] ###########

ax = axs["D"]

ax.set_position([1.5/6 + x_margin, 1/3 + x_margin, 1/6 - 2*x_margin, 1/3 - 2*y_margin])

# ax.spines["top"].set_visible(true)
# ax.spines["right"].set_visible(true)
# ax.spines["left"].set_visible(true)
# ax.spines["bottom"].set_visible(true)

# get results from μ=0.5
x = df[df.μ .== 0.5, :x][1]|>params

plot_network(x, g; ax=ax, clims=[-3.5, 2.1],
                   bias=false, put_colorbar=false)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
for spine in ["top", "right", "left", "bottom"]
    ax.spines[spine].set_visible(true)
end


########### E: split solution μ ∉ [0, 1] ###########

ax = axs["E"]

ax.set_position([1.5/6 + x_margin, x_margin, 1/6 - 2*x_margin, 1/3 - 2*y_margin])

# get results from μ=-1.0
x = df[df.μ .== -1.0, :x][1]|>params

plot_network(x, g; ax=ax, clims=[-3.5, 2.1],
                   bias=false, put_colorbar=false)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
for spine in ["top", "right", "left", "bottom"]
    ax.spines[spine].set_visible(true)
end

######################################################
######################################################
######################################################
######################################################
############      3D SURFACE PLOT     ################
######################################################
######################################################
######################################################
######################################################

println("plotting loss surface")

ax = axs["F"]
ax.set_position([2.5/6, 1.5/3, 2/6, 1.5/3])
ax.axis("off")

net = Net(layers=((k, g, false), (1, identity, false)), input=xx, target=yy, derivs=2)
res = 101  # res^2 points in the grid
lim = 3e-4
zz = zeros(res+1, res+1)  # loss surface
prj_old = nothing  

for (i, μ) in -1:3/res:2 |> enumerate  # loop over μ ∈ [-1, 2]
    global prj_old
    xs = split(x_before_split|>params, split_id, μ)  # non-perturbed point in the manifold
    λs, vs = hessian(net, xs) |> eigen
    λs_filter = λs
    λs_filter[λs.|>abs .< 1e-10] .= 1e100
    # plot the smallest non-zero eigenvalue: if there is a negative one plot that, otherwise the smallest positive one (this is arbitrary, one can also plot the largest positive one)
    λmin_idx = λs_filter |> argmin
    prj = vs[:, λmin_idx]
    if prj_old != nothing
        if dot(prj, prj_old) < 0
            prj = -prj
        end
    else
        prj_old = prj
    end
    if i % 20 == 0 print(".") end
    for (j, p) in -lim:2*lim/res:lim |> enumerate
        x = xs + p*prj
        zz[j, i] = loss(net, x)
    end
end

X = -1:3/res:2
Y = -lim:2*lim/res:lim
X, Y = X' .* ones(length(Y)), Y .* ones(length(X))'

# Define the semi-major and semi-minor axes of the ellipse
a = 1.5  # Semi-major axis
b = lim  # Semi-minor axis
# Create an elliptical mask
center_x, center_y = 0.5, 0
mask = abs.((X .- center_x) ./ a).^1.2 .+ abs.((Y .- center_y) ./ b).^1.2 .<= 1
# Apply the mask to the data
zz_masked = copy(zz)
zz_masked[.!mask] .= NaN  # Set values outside the mask to NaN



fig3d, ax = plt.subplots(subplot_kw=Dict("projection" => "3d"), figsize=(2, 1.5))
ax.plot_surface(X, Y, zz_masked, cmap="viridis", edgecolor="none")
# ax.plot(df[df.μ .== -1, "μproj"][1][1:end-1], y_trj1, losses1, color="red", linewidth=1)
# draw red line across the values of μ and at y=0 and loss=the loss of the original net
range_z = maximum(zz) - minimum(zz)
ax.plot(-1:1/res:0, zeros(length(-1:3/res:2)), loss(net, split(x_before_split|>params, split_id, 0.5)), color="orange", linewidth=0.5, zorder=10)
ax.plot(0:1/res:1, zeros(length(-1:3/res:2)), loss(net, split(x_before_split|>params, split_id, 0.5)), color="red", linewidth=0.5, zorder=10)
ax.plot(1:1/res:2, zeros(length(-1:3/res:2)), loss(net, split(x_before_split|>params, split_id, 0.5)), color="orange", linewidth=0.5, zorder=10)
ax.plot([0, 1], [0, 0], loss(net, split(x_before_split|>params, split_id, 0.5)), color="black", linewidth=1, linestyle="none", marker="o", markersize=0.75, zorder=10)
ax.set_xlabel("μ", labelpad=-11)
ax.set_ylabel("Projection", labelpad=-15)
ax.set_zlabel("Loss", labelpad=-15)
ax.set_zscale("log")
# xticks
ax.set_xticks([-1, 0, 1, 2])
ax.tick_params(axis="x", pad=-6)
ax.tick_params(axis="y", pad=-6)
ax.tick_params(axis="z", pad=-6)

# remove zticklabels
ax.set_zticks([])
ax.zaxis.set_major_locator(PyPlot.matplotlib.ticker.NullLocator())
ax.zaxis.set_minor_locator(PyPlot.matplotlib.ticker.NullLocator())
# same for y axis
ax.yaxis.set_ticklabels([])
# remove background
ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false

ax.view_init(elev=24, azim=53)

fig3d.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
fig3d.savefig(datadir("proc", exp_name, "loss_surface.pdf"), pad_inches=0)
######################################################
######################################################
######################################################
######################################################
############      TRAJECTORY PLOT        #############
######################################################
######################################################
######################################################
######################################################

println("plotting trajectory")

ax = axs["G"]
x_margin = 0.0875
y_margin = 0.12
ax.set_position([2.5/6 + x_margin, y_margin, 2/6 - x_margin, 1.5/3 - y_margin])

d_in = size(x_before_split["w1"])[2]

# the μline is defined as the space of parameters where neuron duplication can occur.
# in particular for two split neuron output weights a₁ and a₂, μa₁ + (1-μ)a₂ = â.
# where â is the output weight before splitting.
# parameter space vector that defines the μline:
μvector = [zeros(d_in*k)..., zeros(split_id-1)..., 1., zeros(k-split_id-1)..., -1.]
# adding â*μ*μvector to a point on the μline, moves this point along the μline by μ.
â = x_before_split["w2"][split_id] 

# vector that defines the point on the μline at μ=0
xs_0 = split(x_before_split|>params, split_id, 0.0)
xs_0 = [xs_0.w1..., xs_0.w2...]

for (i, row) in enumerate(eachrow(df))
    distances = []  # L2 distance to μline
    μprojections = []  # μ value of the projection on μline
    for t in row.trj
        # extract parameters from the trajectory point
        p = t|>params
        p = [p.w1..., p.w2...]
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
        # any point x on the μline is parameterised as follows: x = xs_0 + μ * â * μvector
        # to extract the μ value closest to p we compute the difference vector p-xs_0 and project it on μvector
        μproj = (p-xs_0)'μvector / (2*â)  # the 2 comes from the fact that μ appears twice in the μline equation.
        # the point of intersection between the μline and the distance between p and μline is
        μpoint = xs_0 + μproj * â * μvector
        d = norm(p - μpoint)
        push!(μprojections, μproj)
        push!(distances, d)
    end
    # push entire distance into the dataframe
    df[i, :distances] = distances
    df[i, :μproj] = μprojections
end

for (i, row) in df|>eachrow|>enumerate
    dist = row.distances 
    proj = row.μproj
    ax.plot(proj[1], dist[1], label="init", linewidth=0, markersize=2,
        marker="s", color="black", markerfacecolor="black", markeredgewidth=0.5)
    ax.plot(proj[end-1], dist[end-1], label="final", linewidth=0, markersize=2,
            marker="x", color="black", markerfacecolor="None", markeredgewidth=0.5)
    ax.plot(proj[1:end-2], dist[1:end-2], color="black", linewidth=0.5, label="trajectory")
end

ax.set_yscale("log")
ax.set_xlim(-2.5, 3.5)
ax.set_xticks([-2, 0, 1, 3])
ax.set_xlabel("Projection on "*L"\mu"*" line")
ax.set_ylabel("Distance from μ line")
ax.set_yticks([1e-8, 1e-4, 1])
ax.spines["left"].set_bounds([1e-8, 1])
ax.spines["bottom"].set_bounds(-2, 3)
# move left spine
ax.spines["left"].set_position(("axes", -0.03))

# add legend and make sure there are no duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = Dict(zip(labels, handles))
ordered_labels = ["init", "trajectory", "final"]
ordered_by_label = OrderedDict{String, Any}()
for label in ordered_labels
    if haskey(by_label, label)
        ordered_by_label[label] = by_label[label]
    end
end
ax.legend(ordered_by_label|>values, ordered_by_label|>keys, loc="upper left", ncol=3, fontsize=7, frameon=false,
          handletextpad=0.1, columnspacing=0.5, borderpad=0, bbox_to_anchor=(-0.05, 1), bbox_transform=ax.transAxes)


######################################################
######################################################
######################################################
######################################################
###############      LOSS vs. μ PLOT     #############
######################################################
######################################################
######################################################
######################################################

println("plotting loss vs. μ")

ax = axs["H"]
x_margin = 0.07
y_margin = 0.12
ax.set_position([4.5/6+x_margin, 2/3+y_margin/1.5, 1.5/6-x_margin, 1/3-y_margin/1.5])  # x, y, width, height

ax.plot(df.μ, df.ℒ, marker="D", markersize=2, linewidth=0, color="black")
ax.set_ylabel("Loss")
ax.set_xticks([-2, 0, 1, 3])
ax.set_xticklabels(["-2", "0", "1", "3"])
ax.set_yticks([minimum(df.ℒ), maximum(df.ℒ)])
# make the ytick printed with two decimal places
ax.yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f"))
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
# cut the x and y spines at the left bottom
ax.spines["bottom"].set_bounds(-2, 3)
ax.spines["left"].set_bounds(minimum(df.ℒ), maximum(df.ℒ))


######################################################
######################################################
######################################################
######################################################
###############      EIGS PLOT        ################
######################################################
######################################################
######################################################
######################################################

println("plotting eigenvalues")

ax = axs["I"]
ax.set_position([4.5/6+x_margin, 1/3+y_margin/1.5, 1.5/6-x_margin, 1/3-y_margin/1.5])  # x, y, width, height

# fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))

res = 10001
eigvals = zeros(res+1, length(xs)) # length(xs) eigenvalues
eigvecs = zeros(res+1, length(xs), length(xs)) # length(xs) eigenvectors
angles = zeros(res+1, length(xs))  # angles between eigenvectors at different μ
old_vs = nothing
eigen_permutation = 1:length(xs)|>collect  

lims = [-2, 3]
μs = lims[1]:(lims[2]-lims[1])/res:lims[2]

# compute first eigen 
xs = split(x_before_split|>params, split_id, lims[1])
_, first_eigens = hessian(net, xs) |>Symmetric |> eigen

for (i, μ) in μs |> enumerate
    xs = split(x_before_split|>params, split_id, μ)
    λs, vs = hessian(net, xs) |>Symmetric |> eigen
    eigvecs[i, :, :] = vs  # store eigenvectors
    eigvecs[i, :, :] = eigvecs[i, :, eigen_permutation]  # fix permutation to μ=μs[1] order
    if i != 1
        # check and fix consistency of eigenvectors (when eigenvalues cross zero they could swap order)
        for j in 1:length(xs)
            for k in (j+1):length(xs) 
                if dot(eigvecs[i-1, :, j], eigvecs[i, :, k])|>abs > 0.8
                    eigen_permutation[j], eigen_permutation[k] = eigen_permutation[k], eigen_permutation[j]
                    eigvecs[i, :, j], eigvecs[i, :, k] = eigvecs[i, :, k], eigvecs[i, :, j]
                end       
            end
        end
        # fix consistency in sign of eigenvectors
        for j in 1:length(xs)
            if dot(eigvecs[i, :, j], eigvecs[i-1, :, j]) < 0
                eigvecs[i, :, j] *= -1
            end
        end
    end
    eigvals[i, eigen_permutation] = λs 
    # compute angles between first_eigens and current eigenvectors
    for j in 1:length(xs)
        angles[i, j] = acos(clamp(dot(first_eigens[:, j], eigvecs[i, :, j]), -1, 1))
    end
    if i % 100 == 0 print(".") end
end

for i in 1:length(xs)
    ax.plot(μs, eigvals[:, i], label="λ$i", linewidth=1)
end
ax.set_ylabel("Eigenvalues")
ax.set_yscale("symlog")
ax.set_ylim(-1e-1, 2e-1)
ax.yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f"))
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
ax.spines["bottom"].set_bounds(lims)
ax.set_xticks([-2, 0, 1, 3])
ax.set_xticklabels(["-2", "0", "1", "3"])


fig.savefig(datadir("proc", exp_name, "fig2_eigenvalues.pdf"))

# TODO: consider to break the y-axis and show higher levels (from 0.1 to 10)


######################################################
######################################################
######################################################
######################################################
#############     EIGS ANGLE PLOT     ################
######################################################
######################################################
######################################################
######################################################

println("plotting eigenvector angles")

ax = axs["L"]

ax.set_position([4.5/6+x_margin, y_margin, 1.5/6-x_margin, 1/3-y_margin])  # x, y, width, height

# recomputing with respect to μ=0.5
res = 10001
eigvals = zeros(res+1, length(xs)) # length(xs) eigenvalues
eigvecs = zeros(res+1, length(xs), length(xs)) # length(xs) eigenvectors
angles = zeros(res+1, length(xs))  # angles between eigenvectors at different μ

# compute first eigen 
xs = split(x_before_split|>params, split_id, 0.5)
_, first_eigens = hessian(net, xs) |>Symmetric |> eigen

for (l_idx, lims) in enumerate([[0.5, 3], [0.5, -2]])
    μs = lims[1]:(lims[2]-lims[1])/(res/2):lims[2]
    old_vs = nothing
    eigen_permutation = 1:length(xs)|>collect  
    for (i, μ) in μs |> enumerate
        if l_idx == 1
            i += (res+1)/2|>Int; backwards = -1
        else
            i = ((res+1)/2|>Int) - i + 1; backwards = 1
        end
        xs = split(x_before_split|>params, split_id, μ)
        λs, vs = hessian(net, xs) |>Symmetric |> eigen
        eigvecs[i, :, :] = vs  # store eigenvectors
        eigvecs[i, :, :] = eigvecs[i, :, eigen_permutation]  # fix permutation to μ=μs[1] order
        if i != 1
            # check and fix consistency of eigenvectors (when eigenvalues cross zero they could swap order)
            for j in 1:length(xs)
                for k in (j+1):length(xs) 
                    if dot(eigvecs[i+backwards, :, j], eigvecs[i, :, k])|>abs > 0.8
                        eigen_permutation[j], eigen_permutation[k] = eigen_permutation[k], eigen_permutation[j]
                        eigvecs[i, :, j], eigvecs[i, :, k] = eigvecs[i, :, k], eigvecs[i, :, j]
                    end       
                end
            end
            # fix consistency in sign of eigenvectors
            for j in 1:length(xs)
                if dot(eigvecs[i, :, j], eigvecs[i+backwards, :, j]) < 0
                    eigvecs[i, :, j] *= -1
                end
            end
        end
        eigvals[i, eigen_permutation] = λs 
        # compute angles between first_eigens and current eigenvectors
        for j in 1:length(xs)
            angles[i, j] = acos(clamp(dot(first_eigens[:, j], eigvecs[i, :, j]), -1, 1))
        end
        if i % 100 == 0 print(".") end
    end
end

# TODO: check if angles above μ=0.5 and below are diverging or converging
# check if angles above μ=0.5 and below are diverging or converging
# if they are diverging, then the angle between μ=0 and μ=1 is roughly double the angle between μ=0 and μ=0.5

μs = -2:5/res:3
# TODO: there's a bug with the first angle computed, probably some permutation problem or the addre
μs = μs[2:end]  
angles = angles[2:end, 1:end-3]

for i in 1:(length(xs)-3)
    ax.plot(μs, angles[:, i], label="angle $i", linewidth=1)
end
ax.set_ylabel("Angle")
ax.set_xlabel(L"$\mu$")
ax.set_ylim(-0.05, maximum(angles))
ax.set_yticks([0, π/2])
ax.set_yticklabels(["0", "π/2"])
ax.spines["left"].set_bounds(0, π/2)
ax.spines["bottom"].set_bounds(lims)
ax.set_xticks([-2, 0, 1, 3])
ax.set_xticklabels(["-2", "0", "1", "3"])
# move bottom spine down by 0.05
ax.spines["bottom"].set_position(("axes", -0.03))

########### save figure ###########

fig.set_size_inches(6, 3);  
fig.savefig(datadir("proc", exp_name, "fig2.pdf"))
