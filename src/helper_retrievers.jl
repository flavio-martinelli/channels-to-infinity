#=
 # Created on Wed Jan 8, 2025
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: collection of functions to process results
 #
=#

using MLPGradientFlow
include(srcdir("helper_analyse.jl"))

#--> Retrieve functions
function retrieve_x(res) return (x=res["x"], ode_x=res["ode_x"]) end
function retrieve_init(res) return (init=res["init"],) end
function retrieve_net(res) return (net=netfromres(res),) end
function retrieve_loss(res) return (loss=res["loss"], ode_loss=res["ode_loss"]) end
function retrieve_stopped_by(res) 
    (optim_stopped_by=res["optim_stopped_by"],
     ode_stopped_by=res["ode_stopped_by"]) 
end
function retrieve_losses(res) 
    (loss=res["loss"],
     ode_loss=res["ode_loss"]) 
end
function retrieve_gradient(res)
    net, x = netfromres(res)
    (grad=gradient(net, x),)
end
function retrieve_gnorm(res)
    (gnorm=res["gnorm"], gnorm_regularized=res["gnorm_regularized"])
end
function retrieve_eigs(res)  # risk of segfaulting...
    net, _ = netfromres(res)
    x = retrieve_x(res).x |> params
    eigvals, eigvecs = hessian(net, x, maxnorm=1e3)|>eigen
    (eigvals=eigvals, eigvecs=eigvecs)
end
function retrieve_weight_norms(res)
    x = retrieve_x(res).x
    layer_norms = [weightnorm(x[k]|>params) for k in keys(x)]
    _names = Symbol.(["w$(k[2])norm" for k in keys(x)])
    norms = (wnorm = weightnorm(x|>params),)
    merge(norms, NamedTuple(zip(_names, layer_norms)))
end
function generate_retrieve_closest_pair(layer, metric)
    # returns a function that retrieves the closest pair of rows of the weight matrix of the layer
    function retrieve_closest_pair(res)
        x = retrieve_x(res).x
        _key = Symbol("min_$(metric)_$(layer)")
        if !haskey(x, layer)
            return NamedTuple(zip([_key], [nothing]))
        end
        min_dist, pair = closest_pair(x, metric, layer)
        NamedTuple(zip([_key, Symbol("pair_$(metric)_$(layer)")], [min_dist, pair]))
    end
    retrieve_closest_pair
end
function generare_retrieve_closest_triplet(layer, metric)
    # returns a function that retrieves the closest triplet of rows of the weight matrix of the layer
    function retrieve_closest_triplet(res)
        x = retrieve_x(res).x
        _key = Symbol("min_triplet_$(metric)_$(layer)")
        if !haskey(x, layer)
            return NamedTuple(zip([_key], [nothing]))
        end
        min_dist, triplet = closest_triplet(x, metric, layer)
        NamedTuple(zip([_key, Symbol("triplet_$(metric)_$(layer)")], [min_dist, triplet]))
    end
    retrieve_closest_triplet
end
function generate_retrieve_layer_cosine_distance(layer="w1")
    # returns a function that retrieves the cosine dissimilarity between the weight vectors of the layer
    function retrieve_layer_cosine_distance(res)
        x = retrieve_x(res).x
        _key = Symbol("cos_matrix_$(layer)")
        if !haskey(x, layer)
            return NamedTuple(zip([_key], [nothing]))
        end
        NamedTuple(zip([_key], [layer_cosine_distance(x[layer])]))
    end
    retrieve_layer_cosine_distance
end
function generate_retrieve_cosine_tuplets_below_thr(layer="w1", thr=1e-4)
    # returns a function that retrieves the pairs of weight vectors of the layer with cosine dissimilarity below thr
    function retrieve_cosine_tuplets_below_thr(res)
        x = retrieve_x(res).x
        _key = Symbol("cos_tuplets_below_thr_$(layer)")
        if !haskey(x, layer)
            return NamedTuple(zip([_key], [nothing]))
        end
        cos_matrix = layer_cosine_distance(x[layer])
        pairs = []
        for i in 1:size(cos_matrix, 1)
            for j in (i+1):size(cos_matrix, 2)
                if cos_matrix[i, j] < thr
                    push!(pairs, (i, j))
                end
            end
        end
        pairs = merge_pairs_to_groups(pairs)
        NamedTuple(zip([_key], [pairs]))
    end
    retrieve_cosine_tuplets_below_thr
end
function generate_retrieve_trajectory(; cut_idx=nothing)
    # returns the weight trajectory of the simulation up to the cut_idx
    function retrieve_trajectory(res)
        trj = collect(res["trajectory"])
        if cut_idx != nothing trj = trj[1:cut_idx] end
        (trajectory=trj,)
    end
    retrieve_trajectory
end
function generate_retrieve_x_from_trajectory(; cut_idx=nothing)
    # returns the weight trajectory of the simulation up to the cut_idx
    function retrieve_x_from_trajectory(res)
        if cut_idx == nothing
            return retrieve_x(res)
        else
            x = collect(res["trajectory"])[cut_idx][2]
            return (x_trj=x,)
        end
    end
    retrieve_x_from_trajectory
end
function generate_retrieve_gnorm_from_trajectory(; cut_idx=nothing)
    # returns the weight trajectory of the simulation up to the cut_idx
    function retrieve_gnorm_from_trajectory(res)
        if cut_idx == nothing
            return retrieve_gnorm(res)
        else
            x = collect(res["trajectory"])[cut_idx][2]
            net = netfromres(res)[1]
            gnorm = norm(gradient(net, x |> params))
            return (gnorm_trj=gnorm,)
        end
    end
    retrieve_gnorm_from_trajectory
end
function generate_retrieve_wnorm_from_trajectory(; cut_idx=nothing)
    # returns the weight trajectory of the simulation up to the cut_idx
    function retrieve_wnorm_from_trajectory(res)
        if cut_idx == nothing
            return retrieve_weight_norms(res)
        else
            x = collect(res["trajectory"])[cut_idx][2]
            wnorm = weightnorm(x|>params)
            return (wnorm_trj=wnorm,)
        end
    end
    retrieve_wnorm_from_trajectory
end
