using POMDPs, POMDPModels
using Distributions, TensorFlow
const tf = TensorFlow
using DeepRL
using Gadfly

include("./.julia/v0.6/DeepQLearning/src/tf_helpers.jl")

# mdp = GridWorld(5, 5)
# mdp.reward_states = [GridWorldState(1,5,false), GridWorldState(5,1,false)]
# mdp.reward_values = [-1.0, 1.0]
# mdp.tprob = 1
# env = MDPEnvironment(mdp)

# POMDPs.convert_s(::Type{A}, s::GridWorldState, mdp::GridWorld) where A<:AbstractArray = Float64[s.x/mdp.size_x, s.y/mdp.size_y, s.done]
# POMDPs.convert_s(::Type{GridWorldState}, s::AbstractArray, mdp::GridWorld) = GridWorldState(s[1]*mdp.size_x, s[2]*mdp.size_y, s[3])

function eval_q(sess, q, env; n_eval=100)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        nsteps = 10
        done = false
        r_tot = 0.0
        step = 0


        obs = reset(env)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= nsteps
            action =  get_action(sess, env, q, obs)
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

function build_loss(env, q, target_q, a, r, done_mask)
    loss, td_errors = nothing, nothing
    variable_scope("loss") do
        term = cast(done_mask, Float32)
        A = one_hot(a, n_actions(env))
        q_sa = sum(A.*q, 2)
        q_samp = r + (1 - term).*discount(env.problem).*maximum(target_q, 2)
        td_errors = q_sa - q_samp
        errors = huber_loss(td_errors)
        # errors = (q_sa - q_samp).^2
        loss = mean(errors)
    end
    return loss, td_errors
end


function huber_loss(x, δ::Float64=1.0)
    mask = abs(x) .< δ
    return mask.*0.5.*x.^2 + (1-mask).*δ.*(abs(x) - 0.5*δ)
end


function build_train_op(loss;scope="active_q", lr = 0.1,
                        grad_clip = true,
                        clip_val = 10.)
    optimizer = train.AdamOptimizer(lr)

    train_var = get_train_vars_by_name("active_q")

    grad_vars = train.compute_gradients(optimizer, loss, train_var)
    clip_grads = grad_vars
    if grad_clip
        clip_grads = [(clip_by_norm(gradvar[1], clip_val), gradvar[2]) for gradvar in grad_vars]
    end
    train_op = train.apply_gradients(optimizer, clip_grads)
    grad_norm = global_norm([g[1] for g in clip_grads])
    return train_op, grad_norm
end

function dqn_train(env, s, a, sp, r, done_mask, train_op, grad_norm, q, update_op, sess, rng, replay, max_steps, eps, batch_size, train_freq, target_update_freq, log_freq, eval_freq)
    obs = reset(env)
    done = false
    step = 0
    max_episode_length = 100
    rtot = 0
    episode_rewards = Float64[0.0]
    saved_mean_reward = NaN
    scores_eval = Float64[]
    logg_mean = Float64[]
    logg_loss = Float64[]
    logg_grad = Float64[]
    for t=1:max_steps
        if rand(rng) > eps
            action = get_action(sess, env, q, obs)
        else
            action = sample_action(env)
        end
        if t < max_steps/2
            eps = 1 - (1 - 0.01)/(max_steps/2)*t # decay
        else
            eps = 0.01
        end
        ai = action_index(env.problem, action)
        op, rew, done, info = step!(env, action)
        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        # println(o, " ", action, " ", rew, " ", done, " ", info) #TODO verbose?
        step += 1
        episode_rewards[end] += rew
        if done || step >= max_episode_length
            obs = reset(env)
            push!(episode_rewards, 0.0)
            done = false
            step = 0
            rtot = 0
        end
        num_episodes = length(episode_rewards)
        mean_100ep_reward = mean(episode_rewards[max(1, length(episode_rewards)-101):end])
        if t%train_freq == 0
            s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
            loss_val, grad_val, _ = run(sess,[loss, grad_norm, train_op], Dict(s => s_batch,
                                     a => a_batch,
                                     sp => sp_batch,
                                     r => r_batch,
                                     done_mask => done_batch))
            # loss_val,  _ = run(sess,[loss, train_op], Dict(s => s_batch,
            #                       a => a_batch,
            #                       sp => sp_batch,
            #                       r => r_batch,
            #                       done_mask => done_batch))

            push!(logg_loss, loss_val)
            push!(logg_grad, grad_val)
        end

        if t%target_update_freq == 0
            run(sess, update_op)
        end

        if t%eval_freq == 0
            push!(scores_eval, eval_q(sess, q, env))
        end

        if t%log_freq == 0
            push!(logg_mean, mean_100ep_reward)
            logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3f | Grad %2.3f",
                             t, max_steps, eps, mean_100ep_reward, loss_val, grad_val)
            println(logg)
        end
    end
    return logg_mean, logg_loss , logg_grad, episode_rewards, scores_eval
end


env = POMDPEnvironment(TestPOMDP((5,5), 4, 6))
# mdp = GridWorld(5, 5)
# env = MDPEnvironment(mdp)


obs_dim = obs_dimensions(env)
n_outs = n_actions(env)

g = Graph()
tf.set_def_graph(Graph())
sess = Session(g)


@tf begin
    s = placeholder(Float32, shape=[-1, obs_dim...])
    a = placeholder(Int32, shape=[-1])
    sp = placeholder(Float32, shape=[-1, obs_dim...])
    r = placeholder(Float32, shape=[-1])
    done_mask = placeholder(Bool, shape=[-1])
end

q = mlp(s, [8], 4, scope="active_q")#dense(flatten(s), n_outs; activation=identity, scope="active_q")
target_q =  mlp(sp, [8], 4, scope="target_q") #dense(flatten(sp), n_outs; activation=identity, scope="target_q")

loss, td_errors = build_loss(env, q, target_q, a, r, done_mask)

# optimizer = train.AdamOptimizer(0.005)
# train_op = train.minimize(optimizer, loss, var_list = get_train_vars_by_name("active_q"))

train_op, grad_norm = build_train_op(loss, lr=0.005, grad_clip=true, clip_val=10.0)
update_op = add_update_target_op("active_q", "target_q")



rng = MersenneTwister(3)
batch_size = 32
max_steps = 10000
train_freq = 4
target_update_freq = 500
log_freq = 100
eval_freq = 1000
eps = 1

replay = ReplayBuffer(env, 1000, batch_size)

populate_replay_buffer!(replay, env, max_pop=200)

run(sess, global_variables_initializer())

replay

s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)

find(r_batch.>=1)

logg_mean, logg_loss , logg_grad, episode_rewards, scores_eval = dqn_train(env, s, a, sp, r, done_mask, train_op, grad_norm,
                                q, update_op, sess, rng, replay,
                                max_steps, eps, batch_size, train_freq,
                       target_update_freq, log_freq, eval_freq)


Gadfly.plot(x=1:length(logg_loss), y=logg_loss, Geom.line)


Gadfly.plot(x=1:length(logg_grad), y=logg_grad, Geom.line)

Gadfly.plot(x=1:length(logg_mean), y=logg_mean, Geom.line)

Gadfly.plot(x=1:length(episode_rewards), y=episode_rewards, Geom.line)


Gadfly.plot(x=1:length(scores_eval), y=scores_eval, Geom.line)

eval_q(sess, q, env, n_eval=100)
