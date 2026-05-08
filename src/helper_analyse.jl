#=
 # Created on Wed Jan 8, 2025
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: helper functions for reading experiment results
 #
=#

using MLPGradientFlow, DataFrames, LinearAlgebra, Logging, Hungarian, Clustering

include(srcdir("helper_train.jl"))

#--> Setup functions to go through results

function setup_results(exp_name; sims_dir = datadir("sims", exp_name))
    proc_dir = datadir("proc", exp_name)
    mkpath(proc_dir)
    # saves a results_$(expname).jld2 file in sims_dir/../
    df_setup = collect_results!(sims_dir, valid_filetypes = [".jld2"],verbose=true)
    df_setup, sims_dir, proc_dir
end

function retrieve_infos(df_setup; retrievers=[], verbose=false)
    ``` retrieve infos from the df_setup: each retriever is a function that takes a single res and returns a NamedTuple 
    ```    
    if verbose print("\nProcessing dataframe") end
    global_logger(ConsoleLogger(stderr, Logging.Warn))  # disable info logging
    df = DataFrame()
    for (i, row) in eachrow(df_setup)|>enumerate
        res = row.res
        row_tuple = NamedTuple(row.setting)
        row_tuple = merge(row_tuple, (path = row.path,))
        for retriever in retrievers
            retrieved_tuple = retriever(res)
            overlap = intersect(keys(row_tuple), keys(retrieved_tuple))
            if !isempty(overlap)
                throw(ArgumentError("Retriever \"$retriever\" trying to add duplicate keys: $overlap"))
            end
            row_tuple = merge(row_tuple, retriever(res))
        end
        # Add combined tuple to DataFrame
        push!(df, row_tuple, cols=:union)
        if (verbose && i % 25 == 0) print(".") end
    end
    global_logger(ConsoleLogger(stderr, Logging.Info))  # re-enable info logging
    println("\nDone.")
    df
end

#--> Useful functions for analysis

weightnorm(x) = sum(abs2, x)/(2*length(x))
function cos_dist(x, y)
    1 - dot(x, y) / (norm(x) * norm(y))
end
function layer_cosine_distance(W)
    # returns the cosine dissimilarity between the rows of W
    cosine_sim = 1 .- (W ./ norm.(eachrow(W))) * (W ./ norm.(eachrow(W)))'
end
function l2_dist(x, y)
    norm(x - y)
end
function closest_pair(x, metric=l2dist, layer="w1")
    # returns the shortest "metric" distance between any two rows of W, useful to detect duplicates
    min_dist = Inf
    W = x[layer]
    closest_pair = (0, 0)
    for i in 1:size(W)[1]
        j = i+1
        while j <= size(W)[1]
            dist = metric(W[i,:], W[j,:])
            if dist < min_dist 
                min_dist=dist
                closest_pair = (i, j)
            end
            j+=1
        end
    end
    min_dist, closest_pair
end
function closest_triplet(x, metric=l2_dist, layer="w1")
    # returns the triplet of rows in W with the shortest average pairwise distance
    min_avg_dist = Inf
    W = x[layer]
    n = size(W)[1]
    best_triplet = (0, 0, 0)
    for i in 1:(n-2)
        for j in (i+1):(n-1)
            for k in (j+1):n
                d1 = metric(W[i, :], W[j, :])
                d2 = metric(W[i, :], W[k, :])
                d3 = metric(W[j, :], W[k, :])
                avg_dist = (d1 + d2 + d3) / 3
                if avg_dist < min_avg_dist
                    min_avg_dist = avg_dist
                    best_triplet = (i, j, k)
                end
            end
        end
    end
    min_avg_dist, best_triplet
end

function merge_pairs_to_groups(pairs)
    groups = []
    for (a, b) in pairs
        merged = false
        for group in groups
            if a in group || b in group
                push!(group, a)
                push!(group, b)
                merged = true
                break
            end
        end
        if !merged
            push!(groups, Set([a, b]))
        end
    end
    # Merge overlapping groups
    changed = true
    while changed
        changed = false
        new_groups = []
        skip = Set{Int}()
        for i in 1:length(groups)
            if i in skip
                continue
            end
            g1 = groups[i]
            for j in (i+1):length(groups)
                if j in skip
                    continue
                end
                g2 = groups[j]
                if !isempty(intersect(g1, g2))
                    union!(g1, g2)
                    push!(skip, j)
                    changed = true
                end
            end
            push!(new_groups, g1)
        end
        groups = new_groups
    end
    return [collect(g) for g in groups]
end

function group_close_pairs(close_pairs)
    ``` Group close pairs into sets of connected elements
    ```
    # Initialize sets with pair tuples
    groups = [Set(p) for p in close_pairs]
    # Keep merging until no more merges possible
    merged = true
    while merged
        merged = false
        for i in 1:length(groups)
            for j in (i+1):length(groups)
                if !isempty(intersect(groups[i], groups[j]))
                    union!(groups[i], groups[j])  # Merge sets that share elements
                    deleteat!(groups, j)
                    merged = true
                    break
                end
            end
            merged && break
        end
    end
    # Convert sets to tuples
    return [Tuple(sort(collect(g))) for g in groups]
end

function merge_neurons(W, groups)
    # each group(set of indices) of neurons is merged into a single neuron by averaging the input weights and summing the output weights
    # order of neurons is changed: merged neurons are at the beginning, in order of appearance in groups. Other neurons are appended at the end
    W = deepcopy(W)
    w_merged = []
    a_merged = []
    for idxs in groups
        w = mean([W["w1"][i,:] for i in idxs])
        a = sum([W["w2"][:,i] for i in idxs])
        push!(w_merged, w)
        push!(a_merged, a)
    end
    unmerged = setdiff(1:size(W["w1"],1), collect(Iterators.flatten(groups)))
    w = vcat(hcat(w_merged...)', W["w1"][unmerged,:])
    a = hcat(a_merged..., W["w2"][:,unmerged], W["w2"][:,end])
    Dict("w1" => w, "w2" => a) 
end

function cluster_loss_levels(losses; reltol=1e-3)
    loss_clusters = [losses[1]]
    loss_indices = [[1]]
    for (i, l) in enumerate(losses[2:end])
        if isnan(l) continue end
        clustered = false
        for lc in loss_clusters
            if 1-reltol < l/lc < 1+reltol 
                clustered = true 
                push!(loss_indices[findfirst(x->x==lc, loss_clusters)], i+1)
                break 
            end
        end
        if !clustered push!(loss_clusters, l); push!(loss_indices, [i+1]) end
    end
    loss_clusters, loss_indices 
end

function weight_distance_matrix(p1, p2)
    ### p1 and p2 are two parameter vectors of the same size (i.e. res["x"] from the optimization)
    # concatenate input weight vector with output weight for each neuron
    wa1 = [[p1["w1"][k,:]...; p1["w2"][k]] for k in 1:size(p1["w1"])[1]]
    wa2 = [[p2["w1"][k,:]...; p2["w2"][k]] for k in 1:size(p2["w1"])[1]]
    # compute the pairwise distances between the weight vectors
    _, cost = [norm(wa1[i] - wa2[j]) for i in 1:length(wa1), j in 1:length(wa2)] |> hungarian 
    return cost
end

function compute_unique_solutions(p; plot_debug=false)
    ## p must be a list of parameters of the same size (i.e. res["x"] from the optimization)
    # compute the pairwise distances between all solutions
    dists = [weight_distance_matrix(p[i], p[j]) for i in 1:length(p), j in 1:length(p)]
    # simple hierarchical clustering to find how many unique solutions there are
    h = hclust(dists, linkage=:average, uplo=:U)  
    # cut the tree where the biggest log-difference of distances is
    height_diffs = (h.height .+ eps() .+ 1e-8) .|> log10 |> diff
    cut_height = (h.height[argmax(height_diffs)]+h.height[argmax(height_diffs)+1])/2
    if plot_debug
        fig, ax = plt.subplots()
        ax.plot(h.height.|>log10)
        ax.plot(height_diffs)
    end
    cluster_no = cutree(h, h=cut_height) |> unique |> length
end

function netfromres(res)
    # res is a dictionary with the results of the optimization, this function spits out a trainable net
    lspec = map(x -> (x[1], eval(Meta.parse(x[2])), x[3]), res["layerspec"])
    Net(layers = lspec,
        bias_adapt_input = false,
        input = res["input"],
        target = res["target"],
        derivs = 2),
    params(res["init"])
end

function binomial_CI(binary_vec; level=0.95)
    # computes the binomial confidence interval of a binary vector for the estimated frequency of true elements in the vector
    successes = binary_vec|>sum
    n = length(binary_vec)
    p = successes / n
    binomial_test = HypothesisTests.BinomialTest(successes,n,p)
    confidence_interval = HypothesisTests.confint(binomial_test; level = level, tail = :both, method = :wilson)
    return confidence_interval
end

function binomial_CI(p, n; level=0.95)
    # computes the binomial confidence interval for p successes in n trials
    binomial_test = HypothesisTests.BinomialTest(round(p*n), n, p)
    confidence_interval = HypothesisTests.confint(binomial_test; level = level, tail = :both, method = :wilson)
    return confidence_interval
end