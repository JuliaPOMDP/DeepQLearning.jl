using TensorFlow, DeepRL, Parameters, Distributions
const tf = TensorFlow


@with_kw mutable struct RecurrentQNetworkArchitecture
   fc_in::Vector{Int64} = Vector{Int64}[]
   convs::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
   fc_out::Vector{Int64} = Vector{Int64}[]
   lstm_size::Int64 = 32
end


@with_kw mutable struct DeepRecurrentQLearningSolver
    arch::QNetworkArchitecture = RecurrentQNetworkArchitecture()
    lr::Float64 = 0.001
    max_steps::Int64 = 1000
    target_update_freq::Int64 = 500
    batch_size::Int64 = 32
    trace_length = 6
    train_freq::Int64  = 4
    log_freq::Int64 = 100
    eval_freq::Int64 = 100
    num_ep_eval::Int64 = 100
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    double_q::Bool = true
    dueling::Bool = true
    buffer_size::Int64 = 10000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    verbose::Bool = true
end

# placeholder

function build_placeholders(env::MDPEnvironment, trace_length::Int64)
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    # bs x trace_length x dim
    s = placeholder(Float32, shape=[-1, trace_length, obs_dim...])
    a = placeholder(Int32, shape=[-1, trace_length])
    sp = placeholder(Float32, shape=[-1, trace_length, obs_dim...])
    r = placeholder(Float32, shape=[-1, trace_length])
    done_mask = placeholder(Bool, shape=[-1, trace_length])
    trace_mask = placeholder(Int32, shape=[-1, trace_length])
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
        trace_length = 6
        flat_a = reshape(a, (-1))
        flat_r = reshape(r, (-1))
        flat_done_mask = reshape(done_mask, (-1))
        term = cast(flat_done_mask, Float32)
        A = one_hot(flat_a, n_actions(env))
        q_sa = sum(A.*q, 2)
        q_samp = flat_r + (1 - term).*discount(env.problem).*maximum(target_q, 2)
        td_errors = time_mask.*(q_sa - q_samp)
        errors = huber_loss(td_errors)
        loss = sum(importance_weights.*errors)/sum(time_mask)
    end
    return loss, td_errors
end


function build_doubleq_loss(env::MDPEnvironment,
                            q::Tensor,
                            target_q::Tensor,
                            qp::Tensor,
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
        best_a = indmax(qp, 2)
        best_A = one_hot(best_a, n_actions(env))
        target_q_best = sum(best_A.*target_q, 2)
        q_samp = flat_r + (1 - term).*discount(env.problem).*target_q_best
        td_errors = time_mask.*(q_sa - q_samp)
        errors = huber_loss(td_errors)
        loss = sum(importance_weights.*errors)/sum(time_mask)
    end
    return loss, td_errors
end

mutable struct LSTMPolicy <: Policy
    q::Tensor # Q network
    state::LSTMStateTuple
    s::Tensor # placeholder
    state_ph::LSTMStateTuple # hidden state placeholder
    state_val::LSTMStateTuple
    env::AbstractEnvironment
    sess
end

function LSTMPolicy(sess, env, arch, dueling)
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    init_c = zeros(1, arch.lstm_size)
    init_h = zeros(1, arch.lstm_size)
    # bs x trace_length x dim
    s1 = placeholder(Float32, shape=[-1, 1, obs_dim...])
    q1, q1_init, q1_state = build_recurrent_q_network(s1,
                                                   arch.convs,
                                                   arch.fc_in,
                                                   arch.fc_out,
                                                   arch.lstm_size,
                                                   n_actions(env),
                                                   final_activation = identity,
                                                   scope= Q_SCOPE,
                                                   reuse=true,
                                                   dueling = dueling)
    state_val = LSTMStateTuple(init_c, init_h)
    return LSTMPolicy(q1, q1_state, s1, q1_init, state_val, env, sess)
end

function get_action!(policy::LSTMPolicy, o::Array{Float64}, sess) # update hidden state
    # cannot take a batch of observations
    o = reshape(o, (1, 1, size(o)...))
    feed_dict = Dict(policy.s => o, policy.state_ph => policy.state_val)
    q_val, state_val = run(sess, [policy.q, policy.state], feed_dict)
    policy.state_val = state_val
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end

function get_value(policy::LSTMPolicy, o::Array{Float64}, sess) # update hidden state
    # cannot take a batch of observations
    o = reshape(o, (1, 1, size(o)...))
    feed_dict = Dict(policy.s => o, policy.state_ph => policy.state_val)
    q_val, state_val = run(sess, [policy.q, policy.state], feed_dict)
    return q_val
end

function reset_hidden_state!(policy::LSTMPolicy)# could use zero_state from tf
    hidden_size = get(get_shape(policy.state_ph.c).dims[end])
    init_c = zeros(1, hidden_size)
    init_h = zeros(1, hidden_size)
    policy.state_val = LSTMStateTuple(init_c, init_h)
end




function eval_lstm(policy::LSTMPolicy,
                env::Union{MDPEnvironment, POMDPEnvironment},
                sess;
                n_eval::Int64=100,
                max_episode_length::Int64=100)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset_hidden_state!(lstm_policy)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= max_episode_length
            action = get_action!(lstm_policy, obs, sess)
            # println(action)
            obs, rew, done, info = step!(env, action)
            # println("state ", env.state, " action ", a)
            # println("Reward ", rew)
            # println(obs, " ", done, " ", info, " ", step)
            r_tot += rew
            step += 1
        end
        avg_r += r_tot
        # println(r_tot)

    end
    return  avg_r /= n_eval
end

# include experience replay

mdp = TestMDP((5,5), 1, 6)

env = MDPEnvironment(mdp)

# solver = DeepQLearningSolver(max_steps=10000, lr=0.001, eval_freq=1000,num_ep_eval=100,
#                             arch = QNetworkArchitecture(conv=[], fc=[64,32]),
#                             double_q = false, dueling=true)
trace_length = 6 #hp to add to solver
dueling = true
buffer_size = 10000
batch_size = 32
train_start = 300
lr = 0.0005
grad_clip = true
clip_val = 10.0
rng = MersenneTwister(0)
arch = RecurrentQNetworkArchitecture(lstm_size=32, fc_in=[8])
max_steps = 10000
train_freq = 4
log_freq = 100
eval_freq = 100
num_ep_eval = 100
max_episode_length = 100
target_update_freq = 500
eps_fraction = 0.5
eps_end = 0.01
verbose = true
#######################################################################################################
### BUILD GRAPH

sess = init_session()


s, a, sp, r, done_mask, trace_mask, w = build_placeholders(env, trace_length)


q, hq_in, hq_out = build_recurrent_q_network(s,
                              arch.convs,
                              arch.fc_in,
                              arch.fc_out,
                              arch.lstm_size,
                               n_actions(env),
                               final_activation = identity,
                               scope= Q_SCOPE,
                               dueling = dueling)

qp, hqp_in, hqp_out = build_recurrent_q_network(sp,
                               arch.convs,
                               arch.fc_in,
                               arch.fc_out,
                               arch.lstm_size,
                               n_actions(env),
                               final_activation = identity,
                               scope= Q_SCOPE,
                               reuse=true,
                               dueling = dueling)

target_q, target_hq_in, target_hq_out = build_recurrent_q_network(sp,
                                                               arch.convs,
                                                               arch.fc_in,
                                                               arch.fc_out,
                                                               arch.lstm_size,
                                                               n_actions(env),
                                                               final_activation = identity,
                                                               scope= TARGET_Q_SCOPE,
                                                               reuse=false,
                                                               dueling = dueling)

# loss, td_errors = build_loss(env, q, target_q, a, r, done_mask, trace_mask, w)


loss, td_errors = build_doubleq_loss(env, q, target_q, qp, a, r, done_mask, trace_mask, w)



train_op, grad_norm = build_train_op(loss,
                                     lr=lr,
                                     grad_clip=grad_clip,
                                     clip_val=clip_val)

update_op = build_update_target_op(Q_SCOPE, TARGET_Q_SCOPE)
lstm_policy = LSTMPolicy(sess, env, arch, dueling)
# END OF GRAPH BUILDING

############################################################################
# START TRAIN LOOP



buffer_size = 100000
max_steps = 10000
log_freq = 10
train_freq = 4
target_update_freq = 700
eval_freq = 100
replay = EpisodeReplayBuffer(env, buffer_size, batch_size, trace_length)

populate_replay_buffer!(replay, env, max_pop=1000)

replay

run(sess, global_variables_initializer())



obs = reset(env)
reset_hidden_state!(lstm_policy)
done = false
step = 0
rtot = 0
episode = DQExperience[]
sizehint!(episode, max_episode_length)
episode_rewards = Float64[0.0]
saved_mean_reward = NaN
scores_eval = Float64[]
logg_mean = Float64[]
logg_loss = Float64[]
logg_grad = Float64[]
eps = 1.0
weights = ones(batch_size*trace_length)
init_c = zeros(batch_size, arch.lstm_size)
init_h = zeros(batch_size, arch.lstm_size)
grad_val, loss_val = NaN, NaN
for t=1:max_steps
    if rand(rng) > eps
        action = get_action!(lstm_policy, obs, sess)
    else
        action = sample_action(env)
    end
    # update epsilon
    if t < eps_fraction*max_steps
        eps = 1 - (1 - eps_end)/(eps_fraction*max_steps)*t # decay
    else
        eps = eps_end
    end
    ai = action_index(env.problem, action)
    op, rew, done, info = step!(env, action)
    exp = DQExperience(obs, ai, rew, op, done)
    push!(episode, exp)
    obs = op
    step += 1
    episode_rewards[end] += rew
    if done || step >= max_episode_length
        add_episode!(replay, episode)
        episode = DQExperience[] # empty episode
        obs = reset(env)
        reset_hidden_state!(lstm_policy)
        push!(episode_rewards, 0.0)
        done = false
        step = 0
        rtot = 0
    end
    num_episodes = length(episode_rewards)
    avg100_reward = mean(episode_rewards[max(1, length(episode_rewards)-101):end])
    if t%train_freq == 0
        s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
        feed_dict = Dict(s => s_batch,
                         a => a_batch,
                         sp => sp_batch,
                         r => r_batch,
                         done_mask => done_batch,
                         trace_mask => trace_mask_batch,
                         w => weights,
                         hq_in.c => init_c,
                         hq_in.h => init_h,
                         hqp_in.c => init_c,
                         hqp_in.h => init_h,
                         target_hq_in.c => init_c,
                         target_hq_in.h => init_h
                         )
        loss_val, td_errors_val, grad_val, _ = run(sess,[loss, td_errors, grad_norm, train_op],
                                    feed_dict)
        push!(logg_loss, loss_val)
        push!(logg_grad, grad_val)
    end

    if t%target_update_freq == 0
        run(sess, update_op)
    end

    if t%eval_freq == 0
        # save hidden state before
        hidden_state = lstm_policy.state_val
        push!(scores_eval, eval_lstm(lstm_policy, env, sess, n_eval=num_ep_eval,
                                              max_episode_length=max_episode_length))
        # reset hidden state
        lstm_policy.state_val = hidden_state
    end

    if t%log_freq == 0
        push!(logg_mean, avg100_reward)
        if  verbose
            logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3f | Grad %2.3f",
                             t, max_steps, eps, avg100_reward, loss_val, grad_val)
            println(logg)
        end
    end

end



using Gadfly



plot(x=1:1:length(logg_grad), y=logg_grad)

plot(x=1:1:length(logg_loss), y=logg_loss)


plot(x=1:1:length(logg_mean), y=logg_mean)

plot(x=1:1:length(episode_rewards), y=episode_rewards)

plot(x=1:1:length(scores_eval), y=scores_eval)



eval_lstm(lstm_policy,env,
               n_eval=1,
               max_episode_length=100)



###############################################################################################
## RANDOM STUFF
replay = EpisodeReplayBuffer(env, buffer_size, batch_size, trace_length)

populate_replay_buffer!(replay, env, max_pop=100)

replay

s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
mean(trace_mask_batch)
mean(s_batch)
s_batch == sp_batch




s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
feed_dict = Dict(s => s_batch,
                 a => a_batch,
                 sp => sp_batch,
                 r => r_batch,
                 done_mask => done_batch,
                 trace_mask => trace_mask_batch,
                 w => weights,
                 hq_in.c => init_c,
                 hq_in.h => init_h,
                 target_hq_in.c => init_c,
                 target_hq_in.h => init_h
                 )

loss_val = run(sess, loss, feed_dict)

q_val = run(sess, q, feed_dict)

target_q_val = run(sess, target_q, feed_dict)


lstm_policy.state_val


reset_hidden_state!(lstm_policy)

pol_val = get_value(lstm_policy, s_batch[2,1,:,:,:], sess)







s_in = reshape(s_batch[1,1,:,:,:], (1,1,5,5,1))
state_in = LSTMStateTuple(init_c, init_h)

q2_val = run(sess, lstm_policy.q, Dict(lstm_policy.s => s_in, lstm_policy.state_ph => state_in))




replay._experience[1][1].s

replay._experience[1][5].s

replay

reset_batches!(replay) # might not be necessary
sample_indices = sample(replay.rng, 1:replay._curr_size, replay.batch_size, replace=false)
@assert length(sample_indices) == size(replay._s_batch)[1]
for (i, idx) in enumerate(sample_indices)
    ep = replay._experience[idx]
    # randomized start TODO add as an option of the buffer
    ep_start = rand(replay.rng, 1:length(ep))
    t = 1
    for j=ep_start:min(length(ep), replay.trace_length)
        expe = ep[j]
        replay._s_batch[i,t,indices(replay._s_batch)[3:end]...] = expe.s
        replay._a_batch[i,t] = expe.a
        replay._r_batch[i,t] = expe.r
        replay._sp_batch[i,t,indices(replay._sp_batch)[3:end]...] = expe.sp
        replay._done_batch[i,t] = expe.done
        replay._trace_mask[i,t] = 1
        t += 1
    end
end


ep = replay._experience[90]



o = reset(env)
done = false
step = 1
while !done && step < 10
    action = sample_action(env)
    ai = action_index(env.problem, action)
    op, rew, done, info = step!(env, action)
    println("o ", round(o[1], 3), " a ", ai, " op ", round(op[1], 3), " rew ", rew, " sp ", env.state)
    o = op
    step += 1
end
