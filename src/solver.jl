@with_kw mutable struct DeepQLearningSolver
    qnetwork::Any = nothing # intended to be a flux model 
    evaluation_policy::Any = basic_evaluation
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end)
    learning_rate::Float64 = 1e-4
    max_steps::Int64 = 1000
    batch_size::Int64 = 32
    train_freq::Int64 = 4
    eval_freq::Int64 = 500
    num_ep_eval::Int64 = 100
    double_q::Bool = true 
    dueling::Bool = true
    recurrence::Bool = true
    trace_length::Int64 = 40
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    logdir::String = "log"
    save_freq::Int64 = 10000
    log_freq::Int64 = 100
    verbose::Bool = true
end

function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # check reccurence 
    if is_recurrent(m) && !solver.recurrence
        throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
    end
    replay = initialize_replay_buffer(solver, env)
    policy = NNPolicy(env.problem, solver.qnetwork, ordered_actions(env.problem), obs_dimensions(env))
    active_q = solver.qnetwork 
    target_q = deepcopy(solver.qnetwork)
    optimizer = ADAM(Flux.params(active_q), solver.learning_rate)
    # start training
    reset!(policy)
    obs = reset(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    saved_mean_reward = 0.
    scores_eval = 0.
    model_saved = false
    for t=1:solver.max_steps 
        act, eps = exploration(solver.exploration_policy, policy, env, obs, t, solver.rng)
        ai = actionindex(env.problem, act)
        op, rew, done, info = step!(env, act)
        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            obs = reset(env)
            reset!(policy)
            push!(episode_steps, step)
            push!(episode_rewards, 0.0)
            done = false
            step = 0
            rtot = 0
        end
        num_episodes = length(episode_rewards)
        avg100_reward = mean(episode_rewards[max(1, length(episode_rewards)-101):end])
        avg100_steps = mean(episode_steps[max(1, length(episode_steps)-101):end])
        if t%solver.train_freq == 0

            
            loss_val, td_errors, grad_val
        end
    end
    

end



function initialize_replay_buffer(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.recurrent
        replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    elseif solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end


function loss(q_sa, q_targets)
    td = q_sa - q_targets
    l = mean(huber_loss.(td))
    return l, td
end

function batch_train!(solver::DeepQLearningSolver,
                      optimizer, 
                      active_q, 
                      target_q,
                      replay::ReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    q_values = active_q(flatten_batch(s_batch)) # n_actions x batch_size
    q_sa = [q_values[a_batch[i], i] for i=1:batch_size]
    q_sp_max = @view maximum(target_q(flatten_batch(sp_batch)), dims=1)[:]
    q_targets = r_batch .+ (1.0 .- done_batch).*discount(env.problem).*q_sp_max # n_actions x batch_size
    loss_tracked, td_tracked = loss(q_sa, q_targets)
    loss_val = loss_tracked.data
    td_vals = td_tracked.data
    Flux.reset!(active_q)
    Flux.truncate!(active_q)
    Flux.back!(loss_tracked)
    grad_norm = global_norm(params(model))
    opt()
    return loss_val, td_vals, grad_norm
end

