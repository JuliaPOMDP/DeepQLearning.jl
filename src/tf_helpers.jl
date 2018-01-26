###################################################################################################
# Helpers function for Tensorflow.jl to make building models easier
#


import tf: tanh
import tf.nn: sigmoid, zero_state
import tf.nn.rnn_cell: get_input_dim, LSTMStateTuple

########### General Helpers ###################################

"""
Reset the graph and open a new session
"""
function init_session()
    g = Graph()
    tf.set_def_graph(g)
    sess = Session(g)
    return sess
end


"""
returns a list of trainable variables whose name contains name
"""
function get_train_vars_by_name(name::String)
    return [var for var in get_def_graph().collections[:TrainableVariables]
            if contains(tf.get_name(var.var_node), name)]
end

function Base.ndims(A::AbstractTensor)
    length(get_shape(A).dims)
end


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
        if activation == identity #XXX hack because identity returns a tensor of shape unknown
            a = input*fc_W + fc_b
        else
            a = activation(input*fc_W + fc_b)
        end
        a
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
        if num_output > 0
            out = dense(out, num_output, activation=final_activation, scope=scope*"/fc_out", reuse=reuse)
        end
    end
    return out
end


# Modify the methods for LSTM and dynamic_rnn to reuse weights.
# this might break if there are changes in tensorflow.jl , maybe it should be a PR

function (cell::nn.rnn_cell.LSTMCell)(input, state, input_dim=-1; reuse=false)
    N = get_input_dim(input, input_dim) + cell.hidden_size
    T = eltype(state)
    input = Tensor(input)
    X = [input state.h]
    var = 2/N
    local Wi, Wf, Wo, Wg
    tf.variable_scope("Weights", initializer=Normal(0.0, var), reuse=reuse) do
        Wi = get_variable("Wi", [N, cell.hidden_size], T)
        Wf = get_variable("Wf", [N, cell.hidden_size], T)
        Wo = get_variable("Wo", [N, cell.hidden_size], T)
        Wg = get_variable("Wg", [N, cell.hidden_size], T)
    end

    local Bi, Bf, Bo, Bg
    tf.variable_scope("Bias", initializer=tf.ConstantInitializer(0.0), reuse=reuse) do
        Bi = get_variable("Bi", [cell.hidden_size], T)
        Bf = get_variable("Bf", [cell.hidden_size], T)
        Bo = get_variable("Bo", [cell.hidden_size], T)
        Bg = get_variable("Bg", [cell.hidden_size], T)
    end

    # TODO make this all one multiply
    I = sigmoid(X*Wi + Bi)
    F = sigmoid(X*Wf + Bf)
    O = sigmoid(X*Wo + Bo)
    G = tanh(X*Wg + Bg)
    C = state.c.*F + G.*I
    S = tanh(C).*O

    return (S, LSTMStateTuple(C, S))
end

# modify the function from tensorflow.jl to support dynamic shapes and returns the output at all time steps
function dynamic_rnn(cell::nn.rnn_cell.LSTMCell, inputs, sequence_length=nothing;input_dim=nothing, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN", reuse=false)
    if input_dim === nothing
        input_dim = tf.get_shape(inputs, 3)
    end
    #TODO Make this all work with non-3D inputs

    if time_major
        # TODO Do this in a more efficient way
        inputs=permutedims(inputs, [2,1,3])
    end

    num_steps = convert(tf.Tensor{Int64}, tf.shape(inputs)[2])
    if sequence_length === nothing
        # Works around a bug in upstream TensorFlow's while-loop
        # gradient calculation
        sequence_length = num_steps
    end


    initial_data = inputs[:,1,:]
    if initial_state === nothing
        initial_state = zero_state(cell, initial_data, dtype)
    end
    # By **MAGIC** these values end up in `while_output` even when num_steps=1

    # Calculate first output -- we can't trivially default it,
    # because that would require batch_size to be known statically,
    # and not having a fixed batch_size is pretty nice.
    output, state = cell(initial_data, initial_state, input_dim, reuse=reuse)
    # By **MAGIC** these values end up in `while_output` eve when num_steps=1
    # and the while-loop should not logically run
    all_time_output = expand_dims(output, 2) # add time dimension should be bsx1xstate_dim
    time_step = tf.constant(2) #skip the completed first step
    while_output = @tf while time_step ≤ num_steps
        data = inputs[:, time_step, :]
        local new_state
        new_output = output
        # new_all_time_output = all_time_output

        tf.variable_scope(scope) do
            new_output, new_state = cell(data, state, input_dim)
            # Only update output and state for rows that are not yet passed their ends
            have_passed_end = sequence_length .< time_step
            f(old_arg, new_arg) = tf.select(have_passed_end, old_arg, new_arg)
            new_output = tf.struct_map(f, output, new_output)
            new_state = tf.struct_map(f, state, new_state)
            all_time_output = hcat(all_time_output, expand_dims(new_output, 2))
        end

        [time_step=>time_step+1, state=>new_state, output=>new_output, all_time_output=>all_time_output]
    end

    final_state = while_output[2]
    final_output = while_output[3]
    final_all_time_output = while_output[4]
    final_output, final_state, final_all_time_output
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
