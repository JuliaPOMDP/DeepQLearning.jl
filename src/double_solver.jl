@with_kw mutable struct DoubleDeepQLearningSolver <: Solver
    qnetwork_learn::Any = nothing # intended to be a flux model 
    qnetwork_explore::Any = nothing
    learning_rate::Float64 = 1e-4
    max_steps::Int64 = 1000
    batch_size::Int64 = 32
    train_freq::Int64 = 4
    eval_freq::Int64 = 500
    target_update_freq::Int64 = 500
    num_ep_eval::Int64 = 100
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    gamma_explore::Float64 = 0.95
    gamma_learn::Float64 = 1.0
    evaluation_policy::Any = basic_evaluation
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end, 1.0, 1.0, 0.0)
    trace_length::Int64 = 40
    double_q::Bool = false
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    rng::AbstractRNG = MersenneTwister(0)
    logdir::String = ""
    bestmodel_logdir::String = ""
    save_freq::Int64 = 3000
    log_freq::Int64 = 100
    verbose::Bool = true
end

function POMDPs.solve(solver::DoubleDeepQLearningSolver, problem::MDP; resume_model::Bool=false)
    env = MDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env, resume_model=resume_model)
end

function POMDPs.solve(solver::DoubleDeepQLearningSolver, problem::POMDP; resume_model::Bool=false)
    env = POMDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env, resume_model=resume_model)
end

function POMDPs.solve(solver::DoubleDeepQLearningSolver, env::AbstractEnvironment; resume_model::Bool=false)
    # make logdir
    mkpath(solver.logdir)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
    mkpath(solver.bestmodel_logdir)

    replay = initialize_replay_buffer(solver, env)
     

    scores_eval_explore = -Inf
    violations_explore = Inf
    
    scores_eval_learn = -Inf
    violations_learn = Inf
    
    
    active_q_explore = solver.qnetwork_explore
    active_q_learn = solver.qnetwork_learn


    # record evaluation
    eval_rewards_explore = Float64[]
    eval_violations_explore = Float64[]
    eval_timeout_explore = Float64[]
    eval_steps_explore = Float64[]
    eval_t= Float64[]
    
    # record training
    train_loss_explore = Float64[]
    train_td_errors_explore = Float64[]
    train_grad_val_explore = Float64[]
    train_t= Float64[]

    # record evaluation
    eval_rewards_learn = Float64[]
    eval_violations_learn = Float64[]
    eval_timeout_learn = Float64[]
    eval_steps_learn = Float64[]
    
    # record training
    train_loss_learn = Float64[]
    train_td_errors_learn = Float64[]
    train_grad_val_learn = Float64[]

    # resume model or not
    resume_epoch = 0
    if resume_model
        saved = BSON.load(solver.logdir*"qnetwork.bson")
        resume_epoch +=  saved[:epoch]
        Flux.loadparams!(active_q_explore, saved[:qnetwork_explore])
        Flux.loadparams!(active_q_learn, saved[:qnetwork_learn])

        solver.train_start = 0

        eval_saved = BSON.load(solver.logdir*"eval_rewards.bson")
        eval_rewards_explore = eval_saved[:eval_scores_explore]
        eval_timeout_explore = eval_saved[:eval_timeout_explore]
        eval_violations_explore = eval_saved[:eval_violations_explore]
        eval_steps_explore = eval_saved[:eval_steps_explore]
        eval_t = eval_saved[:eval_t]

        train_saved = BSON.load(solver.logdir*"train_records.bson")
        train_loss_explore = train_saved[:train_loss_explore]
        train_td_errors_explore = train_saved[:train_td_errors_explore]
        train_grad_val_explore = train_saved[:train_grad_val_explore]
        train_t = train_saved[:train_t]

        eval_rewards_learn = eval_saved[:eval_scores_learn]
        eval_timeout_learn = eval_saved[:eval_timeout_learn]
        eval_violations_learn = eval_saved[:eval_violations_learn]
        eval_steps_learn = eval_saved[:eval_steps_learn]

        train_loss_learn = train_saved[:train_loss_learn]
        train_td_errors_learn = train_saved[:train_td_errors_learn]
        train_grad_val_learn = train_saved[:train_grad_val_learn]

        println("resume model from $(solver.logdir) at training epoch $(resume_epoch), the learn reward $(saved[:reward_learn]), learn violations $(saved[:violations_learn]), explore reward $(saved[:reward_explore]), explore violations $(saved[:violations_explore])") 
    end
    
    
    
    policy_learn = NNPolicy(env.problem, active_q_learn, ordered_actions(env.problem), length(obs_dimensions(env)))
    policy_explore = NNPolicy(env.problem, active_q_explore, ordered_actions(env.problem), length(obs_dimensions(env)))
    target_q_learn = deepcopy(solver.qnetwork_learn)
    target_q_explore = deepcopy(solver.qnetwork_explore)
    optimizer_learn = ADAM(Flux.params(active_q_learn), solver.learning_rate)
    optimizer_explore = ADAM(Flux.params(active_q_explore), solver.learning_rate)
    
    # start training
    reset!(policy_learn)
    reset!(policy_explore)
    obs = reset(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    


    for t=1:solver.max_steps 
        act, eps, gamma = exploration(solver.exploration_policy, policy_explore, env, obs, t, solver.rng)
        ai = actionindex(env.problem, act)
        op, rew, done, info = step!(env, act)
        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            obs = reset(env)
            reset!(policy_explore)
            reset!(policy_learn)
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
            #hs = hiddenstates(active_q)
            loss_val_explore, td_errors_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore,min_q_val_explore, loss_val_learn, td_errors_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn,min_q_val_learn = batch_train!(solver, env, optimizer_explore, active_q_explore, target_q_explore, optimizer_learn, active_q_learn, target_q_learn, replay)
            #sethiddenstates!(active_q, hs)

            push!(train_loss_learn, loss_val_learn)
            push!(train_td_errors_learn, mean(td_errors_learn))
            push!(train_grad_val_learn, grad_val_learn)
            push!(train_t, t+resume_epoch)

            push!(train_loss_explore, loss_val_explore)
            push!(train_td_errors_explore, mean(td_errors_explore))
            push!(train_grad_val_explore, grad_val_explore)


        end

        if t%solver.target_update_freq == 0
            target_q_learn = deepcopy(active_q_learn)
            target_q_explore = deepcopy(active_q_explore)
        end

        if t%solver.eval_freq == 0
            scores_eval_explore, violations_explore, steps_explore, timeout_explore = evaluation(solver.evaluation_policy, 
                                 policy_explore, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
            scores_eval_learn, violations_learn, steps_learn, timeout_learn = evaluation(solver.evaluation_policy, 
                                 policy_learn, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
            
            # save evaluation records
            push!(eval_rewards_learn, scores_eval_learn)
            push!(eval_violations_learn, violations_learn)
            push!(eval_steps_learn, steps_learn)
            push!(eval_timeout_learn, timeout_learn)
            push!(eval_t, t+resume_epoch)

            push!(eval_rewards_explore, scores_eval_explore)
            push!(eval_violations_explore, violations_explore)
            push!(eval_steps_explore, steps_explore)
            push!(eval_timeout_explore, timeout_explore)
        end

        if t%solver.log_freq == 0
            #TODO log the training perf somewhere (?dataframes/csv?)
            if  solver.verbose
                @printf("previous training %5d, %5d / %5d eps %0.3f |  avgR %1.3f \n", resume_epoch, t, solver.max_steps, eps, avg100_reward)
                @printf("Loss_learn %2.3e | Grad_learn %2.3e | max_q_learn %1.3f | mean_q_learn %1.3f | min_q_learn %1.3f \n", loss_val_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn, min_q_val_learn)
                @printf("Loss_explore %2.3e | Grad_explore %2.3e | max_q_explore %1.3f | mean_q_explore %1.3f | min_q_explore %1.3f \n", loss_val_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore, min_q_val_explore)
            end             
            bson(solver.logdir*"eval_rewards.bson", eval_scores_learn =eval_rewards_learn, eval_timeout_learn=eval_timeout_learn, eval_violations_learn=eval_violations_learn, eval_steps_learn=eval_steps_learn, eval_scores_explore=eval_rewards_explore, eval_timeout_explore=eval_timeout_explore, eval_violations_explore=eval_violations_explore, eval_steps_explore=eval_steps_explore, eval_t=eval_t)
            bson(solver.logdir*"train_records.bson", train_loss_learn=train_loss_learn, train_td_errors_learn=train_td_errors_learn, train_grad_val_learn=train_grad_val_learn, train_loss_explore=train_loss_explore, train_td_errors_explore=train_td_errors_explore, train_grad_val_explore=train_grad_val_explore, train_t=train_t)
        end


        if t > solver.train_start && t%solver.save_freq == 0
            save_model(solver, active_q_explore, active_q_learn, scores_eval_explore, violations_explore, scores_eval_learn, violations_learn, t+resume_epoch)
        end

    end # end training
    
    saved = BSON.load(solver.logdir*"qnetwork.bson")
    Flux.loadparams!(policy_explore.qnetwork, saved[:qnetwork_explore])
    Flux.loadparams!(policy_learn.qnetwork, saved[:qnetwork_learn])
    """
    if solver.verbose
        @printf("Restore model with explore eval reward %1.3f and explore violations %1.3f at explore epoch %d. \n", saved_mean_reward_explore, saved_mean_violations_explore, saved[:epoch_explore])
        @printf("Restore model with learn eval reward %1.3f and learn violations %1.3f at learn epoch %d. \n", saved_mean_reward_learn, saved_mean_violations_learn, saved[:epoch_learn])
    end
    """
    return (policy_explore, policy_learn)
end


function restore_best_model(solver::DoubleDeepQLearningSolver, problem::MDP)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
    env = MDPEnvironment(problem, rng=solver.rng)
    restore_best_model(solver, env)
end

function restore_best_model(solver::DoubleDeepQLearningSolver, problem::POMDP)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
    env = POMDPEnvironment(problem, rng=solver.rng)
    restore_best_model(solver, env)
end

function restore_best_model(solver::DoubleDeepQLearningSolver, env::AbstractEnvironment)
    active_q_explore = solver.qnetwork_explore
    active_q_learn = solver.qnetwork_learn
    policy_explore = NNPolicy(env.problem, active_q_explore, ordered_actions(env.problem), length(obs_dimensions(env)))
    policy_learn = NNPolicy(env.problem, active_q_learn, ordered_actions(env.problem), length(obs_dimensions(env)))
    saved = BSON.load(solver.logdir*"qnetwork.bson")
    Flux.loadparams!(policy_explore.qnetwork, saved[:qnetwork_explore])
    Flux.loadparams!(policy_learn.qnetwork, saved[:qnetwork_learn])
    println("restore model from $(solver.logdir) with reward_explore $(saved[:reward_explore]) and violations_explore $(saved[:violations_explore]) at epoch $(saved[:epoch]) ")
    println("restore model from $(solver.logdir) with reward_learn $(saved[:reward_learn]) and violations_learn $(saved[:violations_learn]) at epoch $(saved[:epoch]) ")
    return (policy_explore, policy_learn)
end

function initialize_replay_buffer(solver::DoubleDeepQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end



function batch_train!(solver::DoubleDeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer, 
                      active_q, 
                      target_q,
                      s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights, gamma::Float64)
    q_values = active_q(s_batch) # n_actions x batch_size
    
    max_q_val = maximum(q_values)
    mean_q_val = mean(q_values)
    min_q_val = minimum(q_values)

    q_sa = [q_values[a_batch[i], i] for i=1:solver.batch_size] # maybe not ideal
    if solver.double_q
        target_q_values = target_q(sp_batch)
        qp_values = active_q(sp_batch)
        # best_a = argmax(qp_values, dims=1) # fails with TrackedArrays.
        # q_sp_max = target_q_values[best_a]
        q_sp_max = vec([target_q_values[argmax(view(qp_values,:,i)), i] for i=1:solver.batch_size])
    else
        q_sp_max = @view maximum(target_q(sp_batch), dims=1)[:]
    end
    q_targets = r_batch .+ (1.0 .- done_batch).*gamma.*q_sp_max 
    td_tracked = q_sa .- q_targets
    loss_tracked = loss(importance_weights.*td_tracked)
    loss_val = loss_tracked.data
    # td_vals = [td_tracked[i].data for i=1:solver.batch_size]
    td_vals = Flux.data.(td_tracked)
    Flux.back!(loss_tracked)
    grad_norm = globalnorm(params(active_q))
    optimizer()
    return loss_val, td_vals, grad_norm, max_q_val, mean_q_val, min_q_val
end

function batch_train!(solver::DoubleDeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer_explore, 
                      active_q_explore, 
                      target_q_explore,
                      optimizer_learn,
                      active_q_learn,
                      target_q_learn,
                      replay::ReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)

    loss_val_explore, td_errors_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore,min_q_val_explore =  batch_train!(solver, env, optimizer_explore, active_q_explore, target_q_explore, s_batch, a_batch, r_batch, sp_batch, done_batch, ones(solver.batch_size), solve.gamma_explore)

    loss_val_learn, td_errors_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn,min_q_val_learn=  batch_train!(solver, env, optimizer_learn, active_q_learn, target_q_learn, s_batch, a_batch, r_batch, sp_batch, done_batch, ones(solver.batch_size), solve.gamma_learn)

    return loss_val_explore, td_errors_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore,min_q_val_explore, loss_val_learn, td_errors_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn,min_q_val_learn
end

function batch_train!(solver::DoubleDeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer_explore, 
                      active_q_explore, 
                      target_q_explore,
                      optimizer_learn,
                      active_q_learn,
                      target_q_learn,
                      replay::PrioritizedReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    loss_val_explore, td_errors_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore,min_q_val_explore =  batch_train!(solver, env, optimizer_explore, active_q_explore, target_q_explore, s_batch, a_batch, r_batch, sp_batch, done_batch, weights,  solver.gamma_explore)
    loss_val_learn, td_errors_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn,min_q_val_learn=  batch_train!(solver, env, optimizer_learn, active_q_learn, target_q_learn, s_batch, a_batch, r_batch, sp_batch, done_batch, weights,  solver.gamma_learn)
    # may need more consideration here!!
    update_priorities!(replay, indices, td_errors_learn)
    return loss_val_explore, td_errors_explore, grad_val_explore, max_q_val_explore, mean_q_val_explore,min_q_val_explore, loss_val_learn, td_errors_learn, grad_val_learn, max_q_val_learn, mean_q_val_learn,min_q_val_learn
end


function save_model(solver::DoubleDeepQLearningSolver, active_q_explore, active_q_learn, scores_eval_explore::Float64, violations_explore::Float64, scores_eval_learn::Float64, violations_learn::Float64, epoch::Int64)
    weights_explore = Tracker.data.(params(active_q_explore))
    weights_learn = Tracker.data.(params(active_q_learn))
    bson(solver.logdir*"qnetwork.bson", qnetwork_explore=weights_explore, qnetwork_learn=weights_learn, epoch=epoch, reward_explore=scores_eval_explore, violations_explore=violations_explore, reward_learn=scores_eval_learn, violations_learn=violations_learn)
    
end

@POMDP_require solve(solver::DoubleDeepQLearningSolver, mdp::Union{MDP, POMDP}) begin 
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
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
