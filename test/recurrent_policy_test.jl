using TensorFlow, DeepRL, Parameters, Distributions
const tf = TensorFlow


## Idea: give up on the dynamic rnn, use fixed rnn and build two networks:
# one that process only one state, one that process a serie of state
# pass sequence length as a placeholder??

@with_kw mutable struct RecurrentQNetworkArchitecture
   fc_in::Vector{Int64} = Vector{Int64}[]
   convs::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
   fc_out::Vector{Int64} = Vector{Int64}[]
   lstm_size::Int64 = 64
end


# placeholder

function build_placeholders(env::MDPEnvironment)
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    # bs x trace_length x dim
    s = placeholder(Float32, shape=[-1,-1, obs_dim...])
    a = placeholder(Int32, shape=[-1, -1])
    sp = placeholder(Float32, shape=[-1,-1, obs_dim...])
    r = placeholder(Float32, shape=[-1, -1])
    done_mask = placeholder(Bool, shape=[-1, -1])
    trace_mask = placeholder(Int64, shape=[-1, -1])
    w = placeholder(Float32, shape=[-1])
    return s, a, sp, r, done_mask, trace_mask, w
end

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

    # retrieve dynamic shapes
    input_shape = tf.shape(inputs)
    input_ndims = ndims(s)
    obs_dim = [get(d) for d in get_shape(s).dims[3:end]]
    input_dims = [input_shape[i] for i=1:input_ndims]

    # assumes the first two dimensions are batch size and trace length
    out = inputs
    flatten_time_dims = vcat(input_dims[1].*input_dims[2], input_dims[3:end]...)

    out = reshape(out, flatten_time_dims)
    out = reshape(out, (-1, obs_dim...)) # should not do anything but get the tensor shape not unknown
    # feed into conv_to_mlp
    out = cnn_to_mlp(out, convs, hiddens_in, 0, scope=scope, reuse=reuse, dueling=false, final_activation=nn.relu)
    #retrieve time dimension
    flat_dim = get(get_shape(out).dims[end])
    non_flat_time_dims = vcat(input_dims[1], input_dims[2], 25)
    out = reshape(out, non_flat_time_dims)

    # build RNN
    rnn_cell = nn.rnn_cell.LSTMCell(lstm_size)
    c = placeholder(Float64, shape=[-1, lstm_size])
    h =  placeholder(Float64, shape=[-1, lstm_size])
    state_in = LSTMStateTuple(c, h)
    last_out, state, out = dynamic_rnn(rnn_cell,
                                       out,
                                       initial_state=state_in,
                                       input_dim = flat_dim,
                                       reuse=reuse,
                                       scope=scope)
    # output with dueling
    flatten_time_dims = vcat(input_dims[1].*input_dims[2], lstm_size)#tf.shape(out)[end])
    out = reshape(out, flatten_time_dims)
    out = reshape(out, (-1, arch.lstm_size)) # should not do anything but get the tensor shape not unknown
    out = cnn_to_mlp(out, [], arch.fc_out, n_actions(env), scope=scope, reuse=reuse, dueling=dueling, scope=scope)
    #END OF Q NETWORK GRAPH
    return out, state_in
end

function build_loss(env::MDPEnvironment,
                    q::Tensor,
                    target_q::Tensor,
                    a::Tensor,
                    r::Tensor,
                    done_mask::Tensor,
                    trace_mask::Tensor,
                    importance_weights::Tensor)
    loss, td_errors = nothing, nothing
    variable_scope("loss") do
        # flatten time dim
        time_mask = reshape(trace_mask, (-1))
        flat_a = reshape(a, (-1))
        flat_r = reshape(r, (-1))
        flat_done_mask = reshape(done_mask, (-1))
        term = cast(flat_done_mask, Float32)
        A = one_hot(flat_a, n_actions(env))
        q_sa = sum(A.*q, 2)
        q_samp = flat_r + (1 - term).*discount(env.problem).*maximum(target_q, 2)
        td_errors = time_mask.*(q_sa - q_samp)
        errors = huber_loss(td_errors)
        loss = mean(importance_weights.*errors)
    end
    return loss, td_errors
end

# include experience replay

mdp = TestMDP((5,5), 1, 6)
env = MDPEnvironment(mdp)

solver = DeepQLearningSolver(max_steps=10000, lr=0.001, eval_freq=1000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[64,32]),
                            double_q = false, dueling=true)
trace_length = 8 #hp to add to solver
dueling = false
buffer_size = 1000
batch_size = 32
train_start = 300
lr = 0.001
grad_clip = true
clip_val = 10.0
rng = MersenneTwister(0)



arch = RecurrentQNetworkArchitecture()

replay = EpisodeReplayBuffer(env, buffer_size, batch_size, trace_length)

populate_replay_buffer!(replay, env, max_pop=1000)



sess = init_session()


s, a, sp, r, done_mask, trace_mask, w = build_placeholders(env)



q, hq = build_recurrent_q_network(s,
                              arch.convs,
                              arch.fc_in,
                              arch.fc_out,
                              arch.lstm_size,
                               n_actions(env),
                               final_activation = identity,
                               scope= Q_SCOPE,
                               dueling = dueling)
# issue with the scope in the while loop
# do not use double q until this is fixed
# hint change @tf while to tf.while_loop
# qp, hqp = build_recurrent_q_network(sp,
#                                arch.convs,
#                                arch.fc_in,
#                                arch.fc_out,
#                                arch.lstm_size,
#                                n_actions(env),
#                                final_activation = identity,
#                                scope= Q_SCOPE,
#                                reuse=true,
#                                dueling = dueling)

target_q, target_hq = build_recurrent_q_network(sp,
                               arch.convs,
                               arch.fc_in,
                               arch.fc_out,
                               arch.lstm_size,
                               n_actions(env),
                               final_activation = identity,
                               scope= TARGET_Q_SCOPE,
                               reuse=false,
                               dueling = dueling)

loss, td_errors = build_loss(env, q, target_q, a, r, done_mask, trace_mask, w)


optimizer = train.AdamOptimizer(lr)

train_var = get_train_vars_by_name(Q_SCOPE)

grad_vars = train.compute_gradients(optimizer, loss, train_var)
clip_grads = grad_vars
if grad_clip
    clip_grads = [(clip_by_norm(gradvar[1], clip_val), gradvar[2]) for gradvar in grad_vars]
end
train_op = train.apply_gradients(optimizer, clip_grads)
grad_norm = global_norm([g[1] for g in clip_grads])

train_op, grad_norm = build_train_op(loss,
                                     lr=lr,
                                     grad_clip=grad_clip,
                                     clip_val=clip_val)

update_op = build_update_target_op(Q_SCOPE, TARGET_Q_SCOPE)

s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
weights = ones(batch_size*trace_length)
init_c = zeros(batch_size, arch.lstm_size)
init_h = zeros(batch_size, arch.lstm_size)

feed_dict = Dict(s => s_batch,
                 a => a_batch,
                 sp => sp_batch,
                 r => r_batch,
                 done_mask => done_batch,
                 trace_mask => trace_mask_batch,
                 w => weights,
                 hq.c => init_c,
                 hq.h => init_h,
                 target_hq.c => init_c,
                 target_hq.h => init_h
                 )


run(sess, global_variables_initializer())


@time q_val = run(sess, q, feed_dict)


@time tq_val = run(sess, target_q, feed_dict)

loss_val, td_errors_val = run(sess, [loss, td_errors], feed_dict)





a = tf.reshape(trace_mask_batch, (-1))
a = reshape(a, (32, 8))
a == trace_mask_batch


run(sess, update_op)


q_var = get_train_vars_by_name(Q_SCOPE)


target_q_var = get_train_vars_by_name(TARGET_Q_SCOPE)












@time qp_val = run(sess, qp, feed_dictp)

q_val ≈ qp_val

all_time_val = reshape(all_time_val, (2,3,64))

debug

mean(out_val)



s_ = reshape(s_val, (6,5,5,1))
s_ = reshape(s_, (6,25))

# test RNN
a = placeholder(Float32, shape=[-1,-1, 4, 6])
flat_dim = get(get_shape(out).dims[end])

# build RNN
rnn_cell = nn.rnn_cell.LSTMCell(32)
c = placeholder(Float64, shape=[-1, arch.lstm_size])
h =  placeholder(Float64, shape=[-1, arch.lstm_size])
state_in = LSTMStateTuple(c, h)
out, state, all_time = dynamic_rnn(rnn_cell, out,initial_state=state_in,
                                input_dim = flat_dim, reuse=reuse)

# feed_dict
a_val = rand(12,13,4,6)
feed_dict = Dict(a=>a_val, c=>zeros(4,64), h=>zeros(4,64))

run(sess, [all_time], feed_dict)







sess = init_session()

tl = placeholder(Int64, shape=[])

a = placeholder(Float32, shape=[-1,-1,5,5])
a_shape = tf.shape(a)

out = a
dim = vcat(a_shape[1].*a_shape[2], a_shape[3], a_shape[4])
out = reshape(out, dim)
out = 2*out
dim2 = vcat(a_shape[1], a_shape[2], a_shape[3], a_shape[4])
out = reshape(out, dim2)


a_val = rand(2,3,5,5)

# b, state = dynamic_rnn(nn.rnn_cell.LSTMCell(64), b)

run(sess, global_variables_initializer())
run(sess, out, Dict(a=>a_val))

function Base.ndims(A::AbstractTensor)
    length(get_shape(A).dims)
end
