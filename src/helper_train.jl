# Network/data setup and training entry-point used by helper_experiment.jl.

using MLPGradientFlow, Random, DrWatson
@quickactivate
include(srcdir("funs_input.jl"))
include(srcdir("funs_teachers.jl"))

function setup(;Din = 2,
                Nsamples = 10^4,
                seed = 123, 
                rng = Xoshiro(seed),
                parameter_rng = rng,
                teacher = random_teacher,
                init_scale = 1,
                input = standard_normal_input,
                f = softplus, 
                biases = true,
                r = 4,
                kwargs...)
    """
    setup(; Din = 2, Nsamples = 10^4, seed = 123, rng = Xoshiro(seed), parameter_rng = rng, teacher = random_teacher, init_scale = 1, input = standard_normal_input, k = 4, r = 8, f = softplus, biases = true)

    Initializes and sets up a neural network model, input data, and target data for training.

    # Arguments
    - `Din::Int = 2`: Input dimension.
    - `Nsamples::Int = 10^4`: Number of samples.
    - `seed::Int = 123`: Random seed for reproducibility.
    - `rng::AbstractRNG = Xoshiro(seed)`: Random number generator.
    - `parameter_rng::AbstractRNG = rng`: Random number generator for parameter initialization.
    - `teacher::Function = random_teacher`: Function to generate input and target data.
    - `init_scale::Real = 1`: Scaling factor for random initialisation.
    - `input::Function = standard_normal_input`: Function to generate input data.
    - `f::Function = softplus`: Activation function for hidden layers.
    - `biases::Bool = true`: Whether to include biases in the layers.
    - `r::Int = 8`: Number of hidden units in each layer (except the output layer).

    # Returns
    - `net::Net`: The initialized neural network model.
    - `x::Vector{Float64}`: Initial parameter vector for the network.
    - `xt`: The target data. The type of `xt` depends on the `teacher` function used.

    # Details
    This function sets up a neural network for training. It generates input and target data using a `teacher` function, initializes the network architecture based on the provided arguments, and initializes the network parameters with a random initialization scaled by `init_scale`.  The `teacher` function is responsible for creating the training data, and its specific behaviour depends on the chosen teacher.
    """
    inp, targ, xt = teacher(; input, Din, Nsamples, rng, parameter_rng, kwargs...)
    net = Net(layers = ([(ri, f, biases) for ri in r]..., (1, identity, biases)),
    input = inp, target = targ, derivs = 2)
    x = init_scale * random_params(parameter_rng, net)
    return net, x, xt
end

