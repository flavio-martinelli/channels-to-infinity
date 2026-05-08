#=
 # Created on Thu Jan 9, 2025
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: collection of plotting functions
 #
=#

using PyPlot, StatsBase, Printf, HypothesisTests
PyPlot.matplotlib.style.use(srcdir("plot_formatting_settings.mplstyle"))

include(srcdir("funs_teachers.jl"))

# --> Plotting stats functions

function get_log_bins(min, max, n)
    """
    Return n logarithmically spaced bins between min and max.
    """
    return 10 .^ range(log10(min), log10(max), length=n+1)
end

function line_histogram(ax, x, bins; color_area=nothing, normalize=false, kwargs...)
    """
    Plot a histogram with horizontal lines at each bin count.
    """
    print(kwargs)
    h = fit(Histogram, x, bins)
    counts_ = h.weights
    if normalize
        counts_ = counts_ ./ sum(counts_)
    end
    edges_ = h.edges[1]
    bin_centers = (edges_[1:end-1] .+ edges_[2:end]) ./ 2

    # plot the histogram
    global previous_count = 0
    for (count, left, right) in zip(counts_, edges_[1:end-1], edges_[2:end])
        if count != 0
            ax.plot([left, right], [count, count]; kwargs...)  # horizontal line at count
            if color_area != nothing
                ax.fill_between([left, right], [count, count], color=color_area, linewidth=0)
            end    
        end
        if previous_count == count == 0 continue end
        ax.plot([left, left], [previous_count, count]; kwargs...)  # vertical line from previous count to current count
        global previous_count = count
    end
    ax.plot([edges_[end], edges_[end]], [previous_count, 0]; kwargs...)
end


function loss_levels(fig, ax, losses, ks, idxs; debug_mode = false)
    """
    Plot individual seed converged at different losses. Stack seeds horizontally based on their loss level.
    """
    x = zeros(size(losses)[1])
    for (i, idx) in enumerate(idxs)  # allocate x values from left to right for each loss level
        # if isnan(k) continue end
        idx = idx[sortperm(ks[idx])]
        x[idx] .= [j for j in 1:length(idx)]
    end

    # Sort the idxs by their length, so that shorter rows will be printed on top of longer rows
    idxs = idxs[sortperm(length.(idxs), rev=true)]
    
    cmap = get_cmap("plasma", (unique(ks)|>length))

    for ids in idxs
        # makes sure that higher ks are plotted after lower k 
        idmap = Int.(zeros(size(ks)[1]))
        idmap[ids] .= 1
        idmap = BitVector(idmap)
        for (i, k) in enumerate(unique(ks))
            if isnan(k) continue end
            yy = losses[ks .== k .&& idmap, :]
            if length(yy) == 0 continue end
            xx = x[ks .== k .&& idmap]
            if maximum(xx) > 90  # cut off the x axis at 100 for better visualization
                xx = xx[xx .<= 90]
                yy = yy[1:length(xx)] 
                ax.text(91, yy[end], "···", fontsize=4, verticalalignment="center", horizontalalignment="left")
            end
            ax.plot(xx, yy, alpha=0.85, label="k=$(Int(k))", marker="D", linestyle="None",
                    markerfacecolor=cmap(i-1), markeredgecolor=(0.5, 0.5, 0.5, 0.5), markeredgewidth=0.05)
        end
    end
    ax.set_yscale("log")
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.set_xlabel("Frequency", labelpad=-8)
    ax.set_ylabel("Loss")

    # format y axis to not show scientific notation
    if !debug_mode
        formatter = PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f")
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)
    end
end

function loss_levels_binary_cat(fig, ax, losses, ks, idxs, categories, labels=["A", "B"], legend_title="Title")
    """
    Plot individual seed converged at different losses. Stack seeds horizontally based on their loss level.
    """
    col1 = [0.8, 0.8, 0.8]
    col2 = [0.4, 0.4, 0.4]

    x = zeros(size(losses)[1])
    for (i, idx) in enumerate(idxs)  # allocate x values from left to right for each loss level
        # if isnan(k) continue end
        idx = idx[sortperm(ks[idx])]
        x[idx] .= [j for j in 1:length(idx)]
    end

    # Sort the idxs by their length, so that shorter rows will be printed on top of longer rows
    idxs = idxs[sortperm(length.(idxs), rev=true)]
    
    for ids in idxs
        # makes sure that higher ks are plotted after lower k 
        idmap = Int.(zeros(size(ks)[1]))
        idmap[ids] .= 1
        idmap = BitVector(idmap)
        for (i, k) in enumerate(unique(ks))
            if isnan(k) continue end
            yy = losses[ks .== k .&& idmap, :]
            if length(yy) == 0 continue end
            xx = x[ks .== k .&& idmap]
            cats = categories[ks .== k .&& idmap]
            # plot points one by one
            for (xxx, yyy, cat) in zip(xx, yy, cats)
                color = cat ? col1 : col2
                label = cat ? labels[1] : labels[2]
                ax.plot(xxx, yyy, alpha=0.85, label=label, marker="D", linestyle="None",
                        markerfacecolor=color, markeredgecolor=(0.5, 0.5, 0.5, 0.5), markeredgewidth=0.05)
            end
        end
    end
    ax.set_yscale("log")
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Loss")
    ax.set_xticklabels([])
    ax.set_yticklabels([], minor=true)
    ax.set_yticklabels([])

    # ticks
    max_freq = maximum(x)
    min_loss = minimum(losses)
    ax.set_xticks([0, 100, max_freq])
    ax.set_yticks([min_loss, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.minorticks_off()  # Remove minor ticks
    ax.spines["bottom"].set_bounds(1, max_freq);
    ax.spines["left"].set_bounds(1.0, min_loss);
    
    # legend
    # make sure to remove duplicate labels from the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for (i, label) in enumerate(labels)
        if label in unique_labels continue end
        push!(unique_labels, label)
        push!(unique_handles, handles[i])
    end
    ax.legend(unique_handles, unique_labels, title=legend_title, loc="upper right", title_fontsize=7, fontsize=6, frameon=false, markerscale=2.5,
              handleheight=0.0, columnspacing=0, handletextpad=-0.4, borderaxespad=0.0, borderpad=0.0, ncol=2, alignment="center")

end


function histogram_unique_solutions(fig, ax, unique_solutions, df)
    """
     plot a histogram of unique solutions, color each bar with the same color code used for the loss levels
    """
    cmap = get_cmap("plasma", (unique(df.r)|>length))
    for (i, (r, n)) in enumerate(zip(unique(df.r), unique_solutions))
        color = cmap(i-1)
        ax.bar(i, n, color=color, edgecolor="black", linewidth=1)
        # write the number of unique solutions on top of the bars
        ax.text(i, n, n, horizontalalignment="center", verticalalignment="bottom", fontsize=8)
    end

    # plot horizontal lines at 2^r for each r
    for (i, r) in enumerate(unique(df.r))
        ax.axhline(3^r, color="black", linestyle="--", linewidth=0.5, alpha=0.25, zorder=-1)
    end

    # graphics
    ax.set_yscale("log")
    ax.set_xticks(1:length(unique(df.r)))
    ax.set_xticklabels([Int(r) for r in unique(df.r)])
    ax.set_xlabel("Network size")
    ax.set_title("# unique minima", y=1.05)
    ax.set_yticks([3^r for r in unique(df.r)])
    ax.set_yticklabels([Int(3^r) for r in unique(df.r)])
    ax.set_ylim(1, maximum([unique_solutions..., 3^maximum(df.r)]))
    ax.minorticks_off()
    ax.spines["right"].set_bounds(0, maximum(unique_solutions))
    fig.tight_layout()
end

# <--

# --> Plotting network outputs functions

function plot_network(W, f; m=0, q=0, ax=:none, clims=[-2, 2], bias=true, put_colorbar=true, white_thr=0.01)
    # computing input-output
    x = -sqrt(3):0.1:sqrt(3)+0.1
    y = -sqrt(3):0.1:sqrt(3)+0.1
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    input = [reshape(X, :) reshape(Y, :)]
    Din = size(W.w1)[2]-1
    k = size(W.w1)[1]
    _, z, _ = compute_network(W, f, input', Din, k, bias=bias)
    if m != 0 z .+= ([dot(m, row) for row in eachrow(input)] .+ q)' end  # adding linear term
    Z = reshape(z, Int(sqrt(length(z))), Int(sqrt(length(z))))
    # extracting hyperplanes and arrows
    arrow_x = []  # start and end of arrow
    arrow_y = []  # start and end of arrow
    hyperplanes = []  # function if line is not vertical, otherwise scalar of x position
    if bias
        for (wx, wy, b) in eachrow(W.w1)
            """ Solves the linear system in 3 unknowns to find the arrow-base point: t, x, y
            x = t * w1  # feature vec
            y = t * w2  # feature vec
            w1.x + w2.y + b = 0  # kline
            """
            t = -b / norm([wx wy])^2
            push!(arrow_x, [t*wx, .3*wx/norm([wx wy])])
            push!(arrow_y, [t*wy, .3*wy/norm([wx wy])])
            push!(hyperplanes, wy != 0 ? x -> -(b.+wx*x)./wy : -b/wx)
        end
    else   
        for (wx, wy) in eachrow(W.w1)
            push!(arrow_x, [0, .3*wx/norm([wx wy])])
            push!(arrow_y, [0, .3*wy/norm([wx wy])])
            push!(hyperplanes, wy != 0 ? x -> -wx*x/wy : 0)
        end
    end

    #plotting
    if clims == :auto 
        clims = [minimum(Z)-1e-5, maximum(Z)+1e-5]
    end

    xv = [-sqrt(3), 0, sqrt(3)]
    xl = [L"-\sqrt{3}", L"0", L"\sqrt{3}"]
    tv = [clims[1], -1, 0, 1, clims[2]]
    tl = [i < 0 ? "$i" : " $i" for i in tv ]
    
    if ax == :none
        fig, ax = subplots(1,1, figsize=(3,2.5), dpi=250)
    else
        fig = ax.figure
    end

    pp = ax.contourf(x, y, Z, levels=collect(-2.5:0.05:2.5), cmap="viridis",
                     extend="both", vmin=clims[1], vmax=clims[2])
    ax.set_xlim(-sqrt(3), sqrt(3))
    ax.set_ylim(-sqrt(3), sqrt(3))
    ax.set_xticks(xv)
    ax.set_xticklabels(xl)
    ax.set_yticks(xv)
    ax.set_yticklabels(xl)

    if put_colorbar
        cbar = fig.colorbar(pp, ax=ax, ticks=tv, extend="both",
                            format=PyPlot.matplotlib.ticker.FixedFormatter(tl))
        cbar.ax.set_ylim(clims[1], clims[2])
        cbar.solids.set_edgecolor("face")
    end

    for (arrx, arry, hyp, w2) in zip(arrow_x, arrow_y, hyperplanes, W.w2)
        col = -white_thr < w2 < white_thr ? :white : w2>0 ? :red : :blue
        kwargs = Dict(:color=>col, :linestyle=>"-.", :alpha=>0.5, :linewidth=>1.5)
        if typeof(hyp) == Float64
            ax.axvline(hyp, kwargs...)
        else
            ax.plot(x, hyp(collect(x)); kwargs...)
        end
        ax.arrow(arrx[1], arry[1], arrx[2], arry[2], color=col, head_width=0.1, head_length=0.1)
    end
    (fig, ax)
end


function plot_network_grids(df, proc_dir, name_suffix=""; kwargs...)
    for r in df.r |> unique
        print("\nPlotting solutions of r = $r  ")
        df_sub = df[(df.r .== r), :]
        sort!(df_sub, :loss, rev=true)
        r_path = mkpath(joinpath(proc_dir, "nets_$(name_suffix)/r=$(r)"))
        global i = 1
        while i < size(df_sub)[1]
            fig, axes = subplots(6, 4, figsize=(15, 20));
            for (j, ax) in enumerate(axes|>permutedims)
                W = df_sub.x[i] |> params
                plot_network(W, df_sub.f[i]; ax=ax, clims=[-3.75, 2.1], bias=df_sub.biases[i], kwargs...);
                ax.set_title("Id = $(df_sub.seed[i]), L=$(@sprintf("%.2e", df_sub.loss[i]))");
                i += 1
                if i > size(df_sub)[1] 
                    for ax in permutedims(axes)[j+1:end] ax.axis("off") end
                    break
                end
            end
            fig.tight_layout();
            fig.savefig(joinpath(r_path, "$(i-24)_$(i-1).pdf"));
            print(".")  # to keep track of progress
            close(fig);
        end
    end
end

function plot_2d_datascatter(X,Y; ax=:none, clims=[-2,2], xlims=missing, ylims=missing,
                              title="Dataset", s=10, kwargs...)
    """
    Plot a 2D dataset with color coding based on the values in Y.
    :param X: 2D array of shape (2, n_samples)
    :param Y: 1D array of shape (1, n_samples)
    """
    if ax == :none
        fig, ax = subplots(1,1, figsize=(3,2.5), dpi=250)
    else
        fig = ax.figure
    end
    if ismissing(clims)
        clims = [minimum(Y), maximum(Y)]
    end
    if ismissing(xlims)
        xlims = [minimum(X[1, :]), maximum(X[1, :])]
    end
    if ismissing(ylims)
        ylims = [minimum(X[2, :]), maximum(X[2, :])]
    end
    # Plot the data points with color coding 
    p = ax.scatter(X[1, :], X[2, :], c=Y, cmap="viridis", vmin=clims[1], vmax=clims[2], s=s, kwargs...)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(xlims[1], xlims[2])
    ax.set_ylim(ylims[1], ylims[2])
    ax.set_aspect("equal")
    # Add a color bar
    cbar = fig.colorbar(p, ax=ax)
    cbar.set_label("Y values")
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylim(clims[1], clims[2])
    # Set color bar ticks
    cbar.set_ticks([clims[1], (clims[1]+clims[2])/2, clims[2]])
    cbar.set_ticklabels([clims[1], "0", clims[2]])
    # Set color bar label
    cbar.set_label("Y values")
    # Set color bar limits
    cbar.solids.set_clim(clims[1], clims[2])
    fig, ax
end

function count_plot(fig, ax, xlabs, counts; formatter=string, confidence_intervals=nothing)
    """
    Plot a bar plot with the counts of each label.
    """
    ax.bar(xlabs, counts, edgecolor="black", linewidth=1, 
        color=COLOR_MAP3_PASTEL[0b001]) 
    base_offset = 0.03  # 3% of data range
    for i in 1:length(xlabs)
        offset = base_offset * (maximum(counts) - minimum(counts)) 
        if confidence_intervals != nothing
            yerr = (confidence_intervals[xlabs[i]][2] - confidence_intervals[xlabs[i]][1]) / 2 * 100
            y = (confidence_intervals[xlabs[i]][1] + confidence_intervals[xlabs[i]][2]) / 2 * 100
            x = i-1
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=1, alpha=0.5)
            ax.text(xlabs[i], y + yerr + offset, formatter(counts[i]),
            ha="center", va="bottom", fontsize=5)
        else
            ax.text(xlabs[i], counts[i] + offset, formatter(counts[i]),
            ha="center", va="bottom", fontsize=5)
        end    
    end
    ax.set_xticklabels(xlabs, rotation=45, ha="right")
    fig.tight_layout()
    (fig, ax)
end

function uniquefy_legend(ax; kwargs...)
    """
    Remove duplicate labels from the legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for (i, label) in enumerate(labels)
        if label in unique_labels continue end
        push!(unique_labels, label)
        push!(unique_handles, handles[i])
    end
    ax.legend(unique_handles, unique_labels; kwargs...)
end