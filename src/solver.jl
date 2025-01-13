@with_kw mutable struct DeepQLearningSolver{E<:ExplorationPolicy} <: Solver
    qnetwork::Any = nothing # intended to be a flux model
    exploration_policy::E # No default since 9ac3ab
    learning_rate::Float32 = 1f-4
    max_steps::Int64 = 1000
    batch_size::Int64 = 32
    train_freq::Int64 = 4
    eval_freq::Int64 = 500
    target_update_freq::Int64 = 500
    num_ep_eval::Int64 = 100
    double_q::Bool = true
    dueling::Bool = true
    recurrence::Bool = false
    evaluation_policy::Any = basic_evaluation
    trace_length::Int64 = 40
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float32 = 0.6f0
    prioritized_replay_epsilon::Float32 = 1f-6
    prioritized_replay_beta::Float32 = 0.4f0
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    rng::AbstractRNG = MersenneTwister(0)
    logdir::Union{Nothing, String} = "log/"
    save_freq::Int64 = 3000
    log_freq::Int64 = 100
    verbose::Bool = true
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP)
    env = MDPCommonRLEnv{AbstractArray{Float32}}(problem) # ignores solver.rng because CommonRLEnv doesn't have rng support yet
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::POMDP)
    env = POMDPCommonRLEnv{AbstractArray{Float32}}(problem) # ignores solver.rng because CommonRLEnv doesn't have rng support yet
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnv)
    action_map = collect(actions(env))
    action_indices = Dict(a=>i for (i, a) in enumerate(action_map))

    # check reccurence
    if isrecurrent(solver.qnetwork) && !solver.recurrence
        throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
    end
    replay = initialize_replay_buffer(solver, env, action_indices)
    if solver.dueling
        active_q = create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end
    policy = NNPolicy(env, active_q, action_map, length(obs_dimensions(env)))

    return dqn_train!(solver, env, policy, replay)
end

function dqn_train!(solver::DeepQLearningSolver, env::AbstractEnv, policy::AbstractNNPolicy, replay)
    if solver.logdir !== nothing 
        logger = TBLogger(solver.logdir)
        solver.logdir = logger.logdir
    end
    active_q = getnetwork(policy) # shallow copy
    target_q = deepcopy(active_q)
    optimizer = Adam(solver.learning_rate)
    # start training
    resetstate!(policy)
    reset!(env)
    obs = observe(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_steps = Int64[]
    saved_mean_reward = -Inf
    scores_eval = -Inf
    model_saved = false
    eval_next = false
    save_next = false
    action_indices = Dict(a=>i for (i, a) in enumerate(actionmap(policy)))
    for t=1:solver.max_steps
        act = action(solver.exploration_policy, policy, t, obs)
        ai = action_indices[act]
        rew = act!(env, act)
        op = observe(env)
        done = terminated(env)
        exp = DQExperience(obs, ai, Float32(rew), op, done)
        if solver.recurrence
            add_exp!(replay, exp)
        elseif solver.prioritized_replay
            add_exp!(replay, exp, abs(exp.r))
        else
            add_exp!(replay, exp, 0f0)
        end
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length

            if eval_next # wait for episode to end before evaluating
                scores_eval, steps_eval, info_eval = evaluation(solver.evaluation_policy,
                policy, env,
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)
                eval_next = false

                # only save after evaluation
                if save_next
                    model_saved, saved_mean_reward = save_model(solver, active_q, scores_eval, saved_mean_reward, model_saved)
                    save_next = false
                end
                
                if solver.logdir !== nothing 
                    log_value(logger, "eval_reward", scores_eval, step = t)
                    log_value(logger, "eval_steps", steps_eval, step = t)
                    for (k, v) in info_eval
                        log_value(logger, k, v, step = t)
                    end
                end
            end

            reset!(env)
            obs = observe(env)
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
            loss_val, grad_val = batch_train!(solver, env, policy, optimizer, target_q, replay)
            sethiddenstates!(active_q, hs)
        end

        if t % solver.target_update_freq == 0
            target_q = deepcopy(active_q)
        end

        if t % solver.eval_freq == 0
            eval_next = true
        end
        if t % solver.save_freq == 0
            save_next = true
        end

        if t % solver.log_freq == 0 && solver.logdir !== nothing
            nt = loginfo(solver.exploration_policy, t)
            for (k, v) in pairs(nt)
                log_value(logger, String(k), v, step=t)
            end
            if  solver.verbose
                @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e | EvalR %1.3f \n",
                        t, solver.max_steps, nt[1], avg100_reward, loss_val, grad_val, scores_eval)
            end

            log_value(logger, "avg_reward", avg100_reward, step = t)
            log_value(logger, "loss", loss_val, step = t)
            log_value(logger, "grad_val", grad_val, step = t)
        end

    end # end training
    if model_saved
        if solver.verbose
            @printf("Restore model with eval reward %1.3f \n", saved_mean_reward)
        end
        saved_model_state = BSON.load(joinpath(solver.logdir, "qnetwork_state.bson"))[:qnetwork_state]
        Flux.loadmodel!(policy.qnetwork, saved_model_state)
    end
    return policy
end

function initialize_replay_buffer(solver::DeepQLearningSolver, env::AbstractEnv, action_indices)
    # init and populate replay buffer
    if solver.recurrence
        replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    else
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, action_indices, max_pop=solver.train_start)
    return replay #XXX type unstable
end

function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnv,
                      policy::AbstractNNPolicy,
                      optimizer,
                      target_q,
                      replay::PrioritizedReplayBuffer;
                      discount=default_discount(env)
                     )

    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, importance_weights = sample(replay)
   
    active_q = getnetwork(policy) 
    p = Flux.params(active_q)

    loss_val = nothing
    td_vals = nothing

    γ = convert(Float32, discount)
    if solver.double_q
        qp_values = active_q(sp_batch)
        target_q_values = target_q(sp_batch)
        best_a = [CartesianIndex(argmax(qp_values[:, i]), i) for i=1:solver.batch_size]
        q_sp_max = target_q_values[best_a]
    else
        q_sp_max = dropdims(maximum(target_q(sp_batch), dims=1), dims=1)
    end
    q_targets = r_batch .+ (1f0 .- done_batch) .* γ .* q_sp_max

    gs = Flux.gradient(p) do 
        q_values = active_q(s_batch)
        q_sa = q_values[a_batch]
        td_vals = q_sa .- q_targets
        loss_val = sum(huber_loss, importance_weights.*td_vals)
        loss_val /= solver.batch_size
    end
    
    grad_norm = globalnorm(p, gs)
    Flux.Optimise.update!(optimizer, p, gs)

    
    if solver.prioritized_replay
        update_priorities!(replay, indices, td_vals)
    end

    return loss_val, grad_norm
end

# for RNNs
function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnv,
                      policy::AbstractNNPolicy,
                      optimizer,
                      target_q,
                      replay::EpisodeReplayBuffer;
                      discount=default_discount(env)
                     )
    active_q = getnetwork(policy)
    s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = DeepQLearning.sample(replay)
    Flux.reset!(active_q)
    Flux.reset!(target_q)

    p = Flux.params(active_q)

    loss_val = nothing
    td_vals = nothing

    γ = convert(Float32, discount)
    q_targets = [zeros(Float32, solver.batch_size) for i=1:solver.trace_length]
    for i=1:solver.trace_length
        if solver.double_q
            qp_values = active_q(sp_batch[i])
            best_a = [CartesianIndex(argmax(qp_values[:, i]), i) for i=1:solver.batch_size]
            target_q_values = target_q(sp_batch[i])
            q_sp_max = target_q_values[best_a]
        else
            q_sp_max = dropdims(maximum(target_q(sp_batch[i]), dims=1), dims=1)
        end
        q_targets[i] .= r_batch[i] .+ (1f0 .- done_batch[i]) .* γ .* q_sp_max
    end

    Flux.reset!(active_q)

    gs = Flux.gradient(p) do 
        loss_val = 0f0
        for i=1:solver.trace_length
            q_values = active_q(s_batch[i])
            q_sa = q_values[a_batch[i]]
            td_vals = q_sa .- q_targets[i]
            loss_val += sum(huber_loss, trace_mask_batch[i].*td_vals)/solver.batch_size
        end
        loss_val /= solver.trace_length
    end

    grad_norm = globalnorm(p, gs)
    Flux.Optimise.update!(optimizer, p, gs)
    return loss_val, grad_norm
end


function save_model(solver::DeepQLearningSolver, active_q, scores_eval::Float64, saved_mean_reward::Float64, model_saved::Bool)
    if scores_eval >= saved_mean_reward
        copied_model = deepcopy(active_q)
        Flux.reset!(copied_model)
        bson(joinpath(solver.logdir, "qnetwork_state.bson"), qnetwork_state=Flux.state(copied_model))
        if solver.verbose
            @printf("Saving new model with eval reward %1.3f \n", scores_eval)
        end
        model_saved = true
        saved_mean_reward = scores_eval
    end
    return model_saved, saved_mean_reward
end

function restore_best_model(solver::DeepQLearningSolver, problem::MDP)
    env = convert(AbstractEnv, problem) # ignores solver.rng because CommonRLEnv doesn't have rng support yet
    restore_best_model(solver, env)
end

function restore_best_model(solver::DeepQLearningSolver, env::AbstractEnv)
    if solver.dueling
        active_q = create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end
    policy = NNPolicy(env, active_q, collect(actions(env)), length(obs_dimensions(env)))
    saved_network_state = BSON.load(solver.logdir*"qnetwork_state.bson")[:qnetwork_state]
    Flux.loadmodel!(getnetwork(policy), saved_network_state)
    Flux.testmode!(getnetwork(policy))
    return policy
end

POMDPLinter.@POMDP_require POMDPs.solve(solver::DeepQLearningSolver, mdp::Union{MDP, POMDP}) begin
    P = typeof(mdp)
    S = POMDPs.statetype(P)
    A = POMDPs.actiontype(P)
    @req POMDPs.discount(::P)
    @req POMDPs.actions(::P)
    as = POMDPs.actions(mdp)
    @req length(::typeof(as))
    if isa(mdp, POMDP)
        O = obstype(mdp)
        @req POMDPs.convert_o(::Type{AbstractArray}, ::O, ::P)
    else
        @req POMDPs.convert_s(::Type{AbstractArray}, ::S, ::P)
    end
    @req POMDPs.reward(::P,::S,::A,::S)
end
