

"""
Build a q network given an architecture of the form
[CONV]-[FC]
with ReLU activation apart from the output layer
"""
function build_q(input::Tensor,
                arch::QNetworkArchitecture,
                env::MDPEnvironment;
                scope::String="",
                reuse::Bool=false,
                dueling::Bool=false)
    return cnn_to_mlp(input, arch.conv, arch.fc, n_actions(env), scope=scope, reuse=reuse, dueling=dueling)
end

"""
Build a recurrent q network given an architecture of the form
[CONV]-[FC]-[LSTM]-[FC]
with ReLU activation apart from the output layer
"""
function build_recurrent_q_network(inputs::Tensor,
                                   convs::Vector{Tuple{Int64, Vector{Int64}, Int64}},
                                   hiddens_in::Vector{Int64},
                                   hiddens_out::Vector{Int64},
                                   lstm_size::Int64,
                                   num_output::Int64;
                                   final_activation = identity,
                                   scope::String= "drqn",
                                   reuse::Bool = false,
                                   dueling::Bool = false)

    # retrieve static dims
    trace_length = get(get_shape(inputs).dims[2])
    obs_dim = obs_dim = [get(d) for d in get_shape(inputs).dims[3:end]]
    # flatten time dim
    out = reshape(inputs, (-1, obs_dim...)) # should not do anything but get the tensor shape not unknown
    # feed into conv_to_mlp
    out = cnn_to_mlp(out, convs, hiddens_in, 0, scope=scope, reuse=reuse, dueling=false, final_activation=nn.relu)
    #retrieve time dimension
    flat_dim = get(get_shape(out).dims[end])
    out = reshape(out, (-1, trace_length, flat_dim))

    # build RNN
    rnn_cell = nn.rnn_cell.LSTMCell(lstm_size)
    c = placeholder(Float32, shape=[-1, lstm_size])
    h =  placeholder(Float32, shape=[-1, lstm_size])
    state_in = LSTMStateTuple(c, h)
    out, state_out = tf.nn.rnn(rnn_cell,
                               out,
                               initial_state=state_in,
                               reuse=reuse,
                               scope=scope)
    out = stack(out, axis=2)
    # output with dueling
    out = reshape(out, (-1, lstm_size))
    out = cnn_to_mlp(out, [], hiddens_out, num_output, scope=scope, reuse=reuse, dueling=dueling, scope=scope)
    #END OF Q NETWORK GRAPH
    return out, state_in, state_out
end
