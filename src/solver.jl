@with_kw mutable struct DeepQLearningSolver <: Solver
    qnetwork::Any = nothing # intended to be a flux model 
    learning_rate::Float64 = 1e-4
    max_steps::Int64 = 1000
    batch_size::Int64 = 32
    train_freq::Int64 = 4
    eval_freq::Int64 = 500
    target_update_freq::Int64 = 500
    num_ep_eval::Int64 = 100
    double_q::Bool = true 
    dueling::Bool = true
    recurrence::Bool = false
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    evaluation_policy::Any = basic_evaluation
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end)
    trace_length::Int64 = 40
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    rng::AbstractRNG = MersenneTwister(0)
    logdir::String = "log/"
    save_freq::Int64 = 3000
    log_freq::Int64 = 100
    verbose::Bool = true
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::POMDP)
    env = POMDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # check reccurence 
    if isrecurrent(solver.qnetwork) && !solver.recurrence
        throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
    end
    replay = initialize_replay_buffer(solver, env)
    if solver.dueling 
        active_q = create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end
    policy = NNPolicy(env.problem, active_q, ordered_actions(env.problem), length(obs_dimensions(env)))
    return dqn_train!(solver, env, policy, replay)
end

function dqn_train!(solver::DeepQLearningSolver, env::AbstractEnvironment, policy::AbstractNNPolicy, replay)
    logger = Logger(solver.logdir)
    active_q = getnetwork(policy) # shallow copy
    target_q = deepcopy(active_q)
    optimizer = ADAM(solver.learning_rate)
    # start training
    resetstate!(policy)
    obs = reset!(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_steps = Int64[]
    saved_mean_reward = -Inf
    scores_eval = -Inf
    model_saved = false
    eval_next = false
    for t=1:solver.max_steps 
        act, eps = exploration(solver.exploration_policy, policy, env, obs, t, solver.rng)
        ai = actionindex(env.problem, act)
        op, rew, done, info = step!(env, act)
        exp = DQExperience(obs, ai, rew, op, done)
        if solver.recurrence 
            add_exp!(replay, exp)
        elseif solver.prioritized_replay
            add_exp!(replay, exp, abs(exp.r))
        else
            add_exp!(replay, exp, 0.0)
        end
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length

            if eval_next # wait for episode to end before evaluating
                scores_eval = evaluation(solver.evaluation_policy, 
                policy, env,                                  
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)
                eval_next = false 

                log_value(logger, "eval_reward", scores_eval, t)
            end

            obs = reset!(env)
            resetstate!(policy)
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
            hs = hiddenstates(active_q)
            loss_val, td_errors, grad_val = batch_train!(solver, env, policy, optimizer, target_q, replay)
            sethiddenstates!(active_q, hs)
        end

        if t%solver.target_update_freq == 0
            weights = Flux.params(active_q)
            Flux.loadparams!(target_q, weights)
        end

        if t%solver.eval_freq == 0
            eval_next = true
        end

        if t%solver.log_freq == 0
            #TODO log the training perf somewhere (?dataframes/csv?)
            if  solver.verbose
                @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e \n",
                        t, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
            end             
            log_value(logger, "epsilon", eps, t)
            log_value(logger, "avg_reward", avg100_reward, t)
            log_value(logger, "loss", loss_val, t)
            log_value(logger, "grad_val", grad_val, t)
        end
        if t > solver.train_start && t%solver.save_freq == 0
            model_saved, saved_mean_reward = save_model(solver, active_q, scores_eval, saved_mean_reward, model_saved)
        end

    end # end training
    if model_saved
        if solver.verbose
            @printf("Restore model with eval reward %1.3f \n", saved_mean_reward)
            saved_model = BSON.load(joinpath(solver.logdir, "qnetwork.bson"))[:qnetwork]
            Flux.loadparams!(getnetwork(policy), saved_model)
        end
    end
    return policy
end

function initialize_replay_buffer(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.recurrence
        replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    else
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end

function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      policy::AbstractNNPolicy,
                      optimizer,
                      target_q,
                      s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    active_q = getnetwork(policy)
    loss_tracked, td_tracked = q_learning_loss(solver, env, active_q, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    loss_val = loss_tracked.data
    td_vals = Flux.data(td_tracked)
    p = params(active_q)
    Flux.back!(loss_tracked)
    grad_norm = globalnorm(p)
    Flux.Optimise._update_params!(optimizer, p)
    return loss_val, td_vals, grad_norm
end

function q_learning_loss(solver::DeepQLearningSolver, env::AbstractEnvironment, active_q, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    q_values = active_q(s_batch)
    q_sa = diag(view(q_values, a_batch, :))
    if solver.double_q
        target_q_values = target_q(sp_batch)
        qp_values = active_q(sp_batch)
        best_a = Flux.onecold(qp_values)
        q_sp_max = diag(view(target_q_values, best_a, :))
    else
        q_sp_max = @view maximum(target_q(sp_batch), dims=1)[:]
    end
    q_targets = r_batch .+ convert(Vector{Float32}, (1.0 .- done_batch).*discount(env.problem)).*q_sp_max 
    td_tracked = q_sa .- q_targets
    loss_tracked = mean(huber_loss, importance_weights.*td_tracked)
    return loss_tracked, td_tracked
end

function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      policy::AbstractNNPolicy,
                      optimizer, 
                      target_q,
                      replay::PrioritizedReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    loss_val, td_vals, grad_norm = batch_train!(solver, env, policy, optimizer, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, weights)
    if solver.prioritized_replay
        update_priorities!(replay, indices, td_vals)
    end
    return loss_val, td_vals, grad_norm
end

# for RNNs
function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      policy::AbstractNNPolicy,
                      optimizer, 
                      target_q,
                      replay::EpisodeReplayBuffer)
    active_q = getnetwork(policy)
    s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = DeepQLearning.sample(replay)
    Flux.reset!(active_q)
    Flux.reset!(target_q)
    loss_tracked = zero(Flux.Tracker.TrackedReal{Float32})
    td_tracked = Vector{Vector{Flux.Tracker.TrackedReal{Float32}}}(undef, solver.trace_length)
    for i=1:solver.trace_length
        loss_tracked_tmp, td_tracked_tmp = q_learning_loss(solver, env, active_q, target_q, s_batch[i], a_batch[i], r_batch[i], sp_batch[i], done_batch[i], trace_mask_batch[i])
        loss_tracked += loss_tracked_tmp 
        td_tracked[i] = td_tracked_tmp
    end
    loss_tracked /= solver.trace_length
    loss_val = Flux.data(loss_tracked)
    td_vals = Flux.data(td_tracked)
    p = params(active_q)
    Flux.back!(loss_tracked)
    grad_norm = globalnorm(p)
    Flux.Optimise._update_params!(optimizer, p)
    return loss_val, td_vals, grad_norm
end


function save_model(solver::DeepQLearningSolver, active_q, scores_eval::Float64, saved_mean_reward::Float64, model_saved::Bool)
    if scores_eval >= saved_mean_reward
        weights = Tracker.data.(params(active_q))
        bson(joinpath(solver.logdir, "qnetwork.bson"), qnetwork=weights)
        if solver.verbose
            @printf("Saving new model with eval reward %1.3f \n", scores_eval)
        end
        model_saved = true
        saved_mean_reward = scores_eval
    end
    return model_saved, saved_mean_reward
end

function restore_best_model(solver::DeepQLearningSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    restore_best_model(solver, env)
end

function restore_best_model(solver::DeepQLearningSolver, problem::POMDP)
    env = POMDPEnvironment(problem, rng=solver.rng)
    restore_best_model(solver, env)
end

function restore_best_model(solver::DeepQLearningSolver, env::AbstractEnvironment)
    if solver.dueling
        active_q = create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end
    policy = NNPolicy(env.problem, active_q, ordered_actions(env.problem), length(obs_dimensions(env)))
    weights = BSON.load(solver.logdir*"qnetwork.bson")[:qnetwork]
    Flux.loadparams!(getnetwork(policy), weights)
    Flux.testmode!(getnetwork(policy))
    return policy
end

@POMDP_require solve(solver::DeepQLearningSolver, mdp::Union{MDP, POMDP}) begin 
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req actionindex(::P, ::A)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)
    if isa(mdp, POMDP)
        O = obstype(mdp)
        @req convert_o(::Type{AbstractArray}, ::O, ::P)
    else
        @req convert_s(::Type{AbstractArray}, ::S, ::P)
    end
    @req reward(::P,::S,::A,::S)
end
