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
    gamma_begin::Float64 = 0.95
    gamma_end::Float64 = 1.0
    gamma_fraction::Float64 = 0.5
    evaluation_policy::Any = basic_evaluation
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end, gamma_begin, gamma_end, gamma_fraction)
    trace_length::Int64 = 40
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
    lambda::Float64 = 0.0
    safety_mode::Bool = false
    watkins::Bool = false
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP; resume_model::Bool=false)
    env = MDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env, resume_model=resume_model)
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::POMDP; resume_model::Bool=false)
    env = POMDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env, resume_model=resume_model)
end

function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment; resume_model::Bool=false)
    # make logdir
    mkpath(solver.logdir)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
    mkpath(solver.bestmodel_logdir)

    # check reccurence 
    if isrecurrent(solver.qnetwork) && !solver.recurrence
        throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
    end
    replay = initialize_replay_buffer(solver, env)
     

    saved_mean_reward = -Inf
    saved_mean_violations = Inf
    scores_eval = -Inf
    model_saved = false
    violations = Inf
    
    
    if solver.dueling 
        active_q = create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end

    # record evaluation
    eval_rewards = Float64[]
    eval_violations = Float64[]
    eval_timeout = Float64[]
    eval_steps = Float64[]
    eval_t = Float64[]
    
    # record training
    train_loss = Float64[]
    train_td_errors = Float64[]
    train_grad_val = Float64[]
    train_t = Float64[]

    # resume model or not
    resume_epoch = 0
    if resume_model
        saved = BSON.load(solver.logdir*"qnetwork.bson")
        resume_epoch +=  saved[:epoch]
        Flux.loadparams!(active_q, saved[:qnetwork])

        saved_best = BSON.load(solver.bestmodel_logdir*"qnetwork.bson")

        solver.train_start = 0

        eval_saved = BSON.load(solver.logdir*"eval_rewards.bson")
        eval_rewards = eval_saved[:eval_scores]
        eval_timeout = eval_saved[:eval_timeout]
        eval_violations = eval_saved[:eval_violations]
        eval_steps = eval_saved[:eval_steps]
        eval_t = eval_saved[:eval_t]

        train_saved = BSON.load(solver.logdir*"train_records.bson")
        train_loss = train_saved[:train_loss]
        train_td_errors = train_saved[:train_td_errors]
        train_grad_val = train_saved[:train_grad_val]
        train_t = train_saved[:train_t]

        println("resume model from $(solver.logdir) at training epoch $(resume_epoch), the best model has reward $(saved_best[:reward]), violations $(saved_best[:violations]) at epoch $(saved_best[:epoch]).")
    end
    
    
    
    policy = NNPolicy(env.problem, active_q, ordered_actions(env.problem), length(obs_dimensions(env)))
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
    


    done=true    
    for t=1:solver.max_steps 
        act, eps, gamma = exploration(solver.exploration_policy, policy, env, obs, t, solver.rng, reset_mask=done)
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
            hs = hiddenstates(active_q)
            loss_val, td_errors, grad_val, max_q_val, mean_q_val,min_q_val = batch_train!(solver, env, optimizer, active_q, target_q, replay, gamma)
            sethiddenstates!(active_q, hs)

            push!(train_loss, loss_val)
            push!(train_td_errors, mean(td_errors))
            push!(train_grad_val, grad_val)
            push!(train_t, t+resume_epoch)



        end

        if t%solver.target_update_freq == 0
            target_q = deepcopy(active_q)
        end

        if t%solver.eval_freq == 0
            scores_eval, violations, steps, timeout = evaluation(solver.evaluation_policy, 
                                 policy, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
            
            # save evaluation records
            push!(eval_rewards, scores_eval)
            push!(eval_violations, violations)
            push!(eval_steps, steps)
            push!(eval_timeout, timeout)
            push!(eval_t, t+resume_epoch)

        end

        if t%solver.log_freq == 0
            #TODO log the training perf somewhere (?dataframes/csv?)
            if  solver.verbose
                @printf("previous training %5d, %5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e | max_q %1.3f | mean_q %1.3f | min_q %1.3f \n", resume_epoch, t, solver.max_steps, eps, avg100_reward, loss_val, grad_val, max_q_val, mean_q_val, min_q_val)
            end             
            bson(solver.logdir*"eval_rewards.bson", eval_scores=eval_rewards, eval_timeout=eval_timeout, eval_violations=eval_violations, eval_steps=eval_steps, eval_t=eval_t)
            bson(solver.logdir*"train_records.bson", train_loss=train_loss, train_td_errors=train_td_errors, train_grad_val=train_grad_val, train_t=train_t)
        end


        if t > solver.train_start && t%solver.save_freq == 0
            model_saved, saved_mean_reward, saved_mean_violations = save_model(solver, active_q, scores_eval, saved_mean_reward, model_saved, violations, saved_mean_violations, t+resume_epoch, solver.safety_mode)
        end

    end # end training
    if model_saved
        if solver.verbose
            saved = BSON.load(solver.logdir*"qnetwork.bson")
            Flux.loadparams!(policy.qnetwork, saved[:qnetwork])
            Flux.testmode!(policy.qnetwork)
            @printf("Restore model with eval reward %1.3f and violations %1.3f at epoch %d. \n", saved_mean_reward, saved_mean_violations, saved[:epoch])

        end
    end
    return policy
end


function restore_best_model(solver::DeepQLearningSolver, problem::MDP)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
    env = MDPEnvironment(problem, rng=solver.rng)
    restore_best_model(solver, env)
end

function restore_best_model(solver::DeepQLearningSolver, problem::POMDP)
    solver.bestmodel_logdir = solver.logdir * "bestmodel/"
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
    saved = BSON.load(solver.logdir*"qnetwork.bson")
    println("$(solver.bestmodel_logdir)")
    Flux.loadparams!(policy.qnetwork, saved[:qnetwork])
    Flux.testmode!(policy.qnetwork)
    println("restore best model from $(solver.bestmodel_logdir) with reward $(saved[:reward]) and violations $(saved[:violations]) at epoch $(saved[:epoch]) ")
    return policy
end

function initialize_replay_buffer(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.recurrence
        replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    elseif solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end


function loss(td)
    l = mean(huber_loss.(td))
    return l
end

function batch_train!(solver::DeepQLearningSolver,
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

function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer, 
                      active_q, 
                      target_q,
                      replay::ReplayBuffer, gamma::Float64)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    return batch_train!(solver, env, optimizer, active_q, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, ones(solver.batch_size), gamma)
end

function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer, 
                      active_q, 
                      target_q,
                      replay::PrioritizedReplayBuffer, gamma::Float64)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    loss_val, td_vals, grad_norm, max_q_val, mean_q_val, min_q_val = batch_train!(solver, env, optimizer, active_q, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, weights, gamma)
    update_priorities!(replay, indices, td_vals)
    return loss_val, td_vals, grad_norm, max_q_val, mean_q_val, min_q_val
end

# for RNNs
function batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      optimizer, 
                      active_q, 
                      target_q,
                      replay::EpisodeReplayBuffer, gamma::Float64)
        
    s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = DeepQLearning.sample(replay)
    q_values = active_q.(s_batch) # vector of size trace_length n_actions x batch_size
    

    q_sa = [zeros(eltype(q_values[1]), solver.batch_size) for i=1:solver.trace_length]
    for i=1:solver.trace_length  # there might be a more elegant way of doing this
        for j=1:solver.batch_size
            if a_batch[i][j] != 0
                q_sa[i][j] = q_values[i][a_batch[i][j], j]
            end
        end
    end



    if solver.double_q
        target_q_values = target_q.(sp_batch)
        Flux.reset!(active_q)
        qp_values = active_q.(sp_batch)
        # best_a = argmax.(qp_values, dims=1)
        # q_sp_max = broadcast(getindex, target_q_values, best_a)
        q_sp_max = [vec([target_q_values[j][argmax(view(qp_values[j],:,i)), i] for i=1:solver.batch_size]) for j=1:solver.trace_length] #XXX find more elegant way to do this
        if solver.watkins
            lambda_sp_batch = [ vec(solver.lambda.*[argmax(view(qp_values[j],:,i))==a_batch[j+1][i]  for i=1:solver.batch_size]) for j=1:solver.trace_length-1]
        else 
            lambda_sp_batch = solver.lambda * [ones(solver.batch_size) for _ in 1:solver.trace_length-1] 
        end
    else
        qp_values = target_q.(sp_batch)
        q_sp_max = vec.(maximum.(qp_values, dims=1))
        if solver.watkins
            lambda_sp_batch = solver.lambda.*[vec([argmax(view(qp_values[j],:,i))==a_batch[j+1][i] for i=1:solver.batch_size]) for j=1:solver.trace_length-1]
        else
            lambda_sp_batch = solver.lambda * [ones(solver.batch_size) for _ in 1:solver.trace_length-1] 
        end
    end

    """
    q_targets = Vector{eltype(q_sa)}(undef, solver.trace_length)
    for i=1:solver.trace_length
        q_targets[i] = r_batch[i] .+ (1.0 .- done_batch[i]).*discount(env.problem).*q_sp_max[i]
    end
    """
    
    lambda_sp_batch = broadcast((x,y) -> x.*y, lambda_sp_batch, trace_mask_batch[2:end])
    q_targets = Vector{Array{Float64}}(undef, solver.trace_length)
    q_targets[end] = r_batch[end]
    denominator = ones(solver.batch_size)

    for i=solver.trace_length-1:-1:1
        q_targets[i] = Flux.data(r_batch[i] .+ (1.0 .- done_batch[i]).* discount(env.problem) .* (q_sp_max[i] .+ Flux.data.(lambda_sp_batch[i].*q_targets[i+1].*denominator)) ./(1.0 .+ denominator .* lambda_sp_batch[i]))
        denominator = 1.0 .+ denominator .* lambda_sp_batch[i]  
    end
    
       
    td_tracked = broadcast((x,y) -> x.*y, trace_mask_batch, q_sa .- q_targets)
    loss_tracked = sum(loss, td_tracked)/solver.trace_length
    loss_val = Flux.data(loss_tracked)

    Flux.reset!(active_q)
    Flux.truncate!(active_q)
    Flux.reset!(target_q)
    Flux.truncate!(target_q)

    td_vals = map(x -> mean(Flux.data.(x)), td_tracked)
    Flux.back!(loss_tracked)
    grad_norm = globalnorm(params(active_q))
    optimizer()



    #q_vals =  broadcast((x,y) -> x.*y, trace_mask_batch, q_sa)
    max_q_val =  maximum(maximum.(q_values))
    mean_q_val = mean(mean.(q_values))
    min_q_val = minimum(minimum.(q_values))
    return loss_val, td_vals, grad_norm, max_q_val, mean_q_val, min_q_val
end

function save_model(solver::DeepQLearningSolver, active_q, scores_eval::Float64, saved_mean_reward::Float64, model_saved::Bool, violations::Float64, saved_mean_violations::Float64, epoch::Int64, safety_mode::Bool)
    weights = Tracker.data.(params(active_q))
    bson(solver.logdir*"qnetwork.bson", qnetwork=weights, epoch=epoch, reward=scores_eval, violations=violations)
    if (safety_mode && (violations <= saved_mean_violations || (violations <= saved_mean_violations+0.1 && scores_eval >= saved_mean_reward))) ||(!safety_mode && scores_eval>=saved_mean_reward)
        saved_mean_reward = scores_eval
        saved_mean_violations = violations
        bson(solver.bestmodel_logdir*"qnetwork.bson", qnetwork=weights, reward=saved_mean_reward, violations=saved_mean_violations, epoch=epoch)
        if solver.verbose
            @printf("Saving new best model with violations %1.3f (%%) and eval reward %1.3f at epoch %d \n", violations, scores_eval, epoch)
        end
        model_saved = true
    end
    return model_saved, saved_mean_reward, saved_mean_violations
end

@POMDP_require solve(solver::DeepQLearningSolver, mdp::Union{MDP, POMDP}) begin 
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
