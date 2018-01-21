###################################################################################################
# Helpers function for Tensorflow.jl to make building models easier
#

########### General Helpers ###################################




############ MODEL BUILDING ###################################

"""
    compute_flat_dim(dims::Vector{Nullable{Int64}})
Multiply all the known dimensions
"""
function compute_flat_dim(dims::Vector{Nullable{Int64}})
    res = 1
    for i in dims
        if isnull(i)
            continue
        end
        res *= get(i)
    end
    return res
end



"""
flatten an array or a tensor and keep the batch size
    flatten(x::Array{Float64}; batch_dim::Int64=1)
    flatten(x::Tensor; batch_dim::Int64=1)
"""
function flatten(x::Array{Float64}; batch_dim::Int64=1)
    x_dims = size(x)
    N = ndims(x)
    batch_size = x_dims[batch_dim]
    if batch_dim != 1 #permute dims such that it is in the first axis
        perm = [mod1(batch_dim + i, N) for i=0:N-1]
        x = permutedims(x, perm)
    end
    flat_dim = div(length(x), batch_size)
    @assert batch_size == size(x)[1]
    x = reshape(x, (batch_size, flat_dim))
    return x
end

function flatten(x::Tensor; batch_dim::Int64=1)
    shape = get_shape(x)
    N = length(shape.dims)
    flat_dim = compute_flat_dim(shape.dims)
    if batch_dim != 1 #permute dims such that it is in the first axis
        perm = [mod1(batch_dim + i, N) for i=0:N-1]
        x = permutedims(x, perm)
    end
    x = reshape(x, (-1, flat_dim))
    return x
end

"""
    Build a fully connected layer followed by an activation function
    returns the output of the activation layer as a tensor
    assumes that the input is of shape (batch size, n)
    uses Xavier initialization
    dense(input::Tensor, hidden_units; activation=identity, scope="")
"""
function dense(input::Tensor, hidden_units; activation=identity, scope="fc", reuse=false)
    input_shape = get(get_shape(input).dims[end])
    weight_shape = (input_shape, hidden_units)
    var = 2/(input_shape + hidden_units) # Xavier initialization
    fc_W = variable_scope(scope, initializer= Normal(0.0, var),  reuse=reuse) do
        get_variable("weight", [weight_shape...], Float32)
    end
    fc_b = variable_scope(scope, initializer=tf.ConstantInitializer(0.),  reuse=reuse) do
        get_variable("bias", [hidden_units], Float32)
    end
    a = variable_scope(scope,  reuse=reuse) do
        a = activation(input*fc_W + fc_b)
    end
    return a
end

"""
    Build a 2d convolutional layer
    returns the output of the activation as a tensor
    Assumes that the input is of shape (batch, height, width, channels)
    uses Xavier initialization
    conv2d(x, W)

    This version also adds an activation layer
    conv2d(inputs::Tensor, num_filters::Int64, kernel_size::Vector{Int64}, activation=identity; stride::Int64=1, padding::String="SAME")

"""
function conv2d(inputs::Tensor, num_filters::Int64, kernel_size::Vector{Int64};
                activation=identity, stride::Int64=1, padding::String="SAME", scope="conv", reuse::Bool=false)
    # assume inputs is of shape batch, height, width, channels
    input_channels = get(get_shape(inputs).dims[end])
    weight_shape = (kernel_size[1], kernel_size[2], input_channels, num_filters)
    var = 1/(kernel_size[1]*kernel_size[2]*input_channels) # Xavier initialization
    conv_W =  variable_scope(scope, initializer=Normal(0.0, var), reuse=reuse) do
        get_variable("weight", [weight_shape...], Float32)
    end
    conv_b = variable_scope(scope, initializer=tf.ConstantInitializer(0.),  reuse=reuse) do
        get_variable("bias", [num_filters], Float32)
    end
    a = variable_scope(scope,  reuse=reuse) do
        z = nn.conv2d(inputs, conv_W, Int64[1, stride, stride, 1], padding) + conv_b
        a = activation(z)
        a = cast(a, Float32)
    end
    return a
end

function conv2d(x, W)
    nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

"""
    Build a max pooling layer, divides the height and width of the input by 2
"""
function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

"""
    Build a multi-layer perceptron (mlp) model given the list of nodes in each layer and the input tensor
"""
function mlp(input::Tensor,
            hiddens::Vector{Int64},
            num_output::Int64;
            final_activation=identity,
            scope="mlp",
            reuse = false
            )
    a = flatten(input, batch_dim=1)
    for (i,h) in enumerate(hiddens)
        a = dense(a, h, activation=nn.relu, scope=scope*"/fc_$i", reuse=reuse)
    end
    a = dense(a, num_output, activation=final_activation, scope=scope*"/fc_out", reuse=reuse)
    return a
end


"""
    Builds a model starting with convolutional layers followed by a multi-layer perceptron
- convs: [(int, (int,int), int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
- hiddens: [int]
        list of sizes of hidden layers

"""

function cnn_to_mlp(inputs, convs, hiddens, num_output;
    final_activation=identity, scope="conv2mlp", reuse=false, dueling=false)
    out = inputs
    for (i,(nfilters, kernel_size, stride)) in enumerate(convs)
        out = conv2d(out, nfilters, kernel_size,
                     activation=nn.relu, stride=stride, scope=scope*"/conv_$i", reuse=reuse)
    end
    out = variable_scope(scope,  reuse=reuse) do
        flatten(out, batch_dim=1)
    end
    if dueling
        state_val_scope = scope*"/state_value"
        state_out = out
        for (i,h) in enumerate(hiddens)
            state_out = dense(state_out, h, activation=nn.relu, scope=state_val_scope*"/fc_$i", reuse=reuse)
        end
        state_out = dense(state_out, 1, activation=final_activation, scope=state_val_scope*"/fc_out", reuse=reuse)

        action_val_scope = scope*"/action_value"
        action_out = out
        for (i,h) in enumerate(hiddens)
            action_out = dense(action_out, h, activation=nn.relu, scope=action_val_scope*"/fc_$i", reuse=reuse)
        end
        action_out = dense(action_out, num_output, activation=final_activation, scope=action_val_scope*"/fc_out", reuse=reuse)

        actions_mean = Ops.expand_dims(mean(action_out, 2),2) # shape bs x 1
        println(get_shape(actions_mean))
        actions_scaled = action_out - actions_mean # broadcast bs x n_actions - bs x 1
        out = state_out + actions_scaled
    else
        for (i,h) in enumerate(hiddens)
            out = dense(out, h, activation=nn.relu, scope=scope*"/fc_$i", reuse=reuse)
        end
        out = dense(out, num_output, activation=final_activation, scope=scope*"/fc_out", reuse=reuse)
    end
    return out
end


"""
returns a list of trainable variables whose name contains name
"""
function get_train_vars_by_name(name::String)
    return [var for var in get_def_graph().collections[:TrainableVariables]
            if contains(tf.get_name(var.var_node), name)]
end

# functions from tensorflow.jl tutorial, no control over the initialization
function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end


function bias_variable(shape)
    initial = zeros(shape...)
    return Variable(initial)
end
