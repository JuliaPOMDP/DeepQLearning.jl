
function POMDPs.solve(solver::DeepQLearningSolver, problem::Union{MDP, POMDP})
    if isa(problem, POMDP) # TODO use multiple dispatch or same as in deepRL.jl?
        env = POMDPEnvironment(problem, rng=solver.rng)
    else
        env = MDPEnvironment(problem, rng=solver.rng)
    end
    #init session and build graph Create a TrainGraph object with all the tensors
    train_graph = build_graph(solver, env)

    # init and populate replay buffer
    if solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    #TODO save the training log somewhere
    avg_r, loss, grad, rewards, eval_r = dqn_train(solver, env, train_graph, replay)
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    return policy, eval_r, avg_r, loss, grad, train_graph
end



function dqn_train(solver::DeepQLearningSolver,
                   env::Union{MDPEnvironment, POMDPEnvironment},
                   graph::TrainGraph,
                   replay::Union{ReplayBuffer, PrioritizedReplayBuffer})
    obs = reset(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    saved_mean_reward = NaN
    scores_eval = Float64[]
    logg_mean = Float64[]
    logg_loss = Float64[]
    logg_grad = Float64[]
    eps = 1.0
    weights = ones(solver.batch_size)
    for t=1:solver.max_steps
        if rand(solver.rng) > eps
            action = get_action(graph, env, obs)
        else
            action = sample_action(env)
        end
        # update epsilon
        if t < solver.eps_fraction*solver.max_steps
            eps = 1 - (1 - solver.eps_end)/(solver.eps_fraction*solver.max_steps)*t # decay
        else
            eps = solver.eps_end
        end
        ai = action_index(env.problem, action)
        op, rew, done, info = step!(env, action)
        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            obs = reset(env)
            push!(episode_rewards, 0.0)
            done = false
            step = 0
            rtot = 0
        end
        num_episodes = length(episode_rewards)
        avg100_reward = mean(episode_rewards[max(1, length(episode_rewards)-101):end])
        if t%solver.train_freq == 0
            if solver.prioritized_replay
                s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
            else
                s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
            end
            feed_dict = Dict(graph.s => s_batch,
                             graph.a => a_batch,
                             graph.sp => sp_batch,
                             graph.r => r_batch,
                             graph.done_mask => done_batch,
                             graph.importance_weights => weights)
            loss_val, td_errors, grad_val, _ = run(graph.sess,[graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                        feed_dict)

            push!(logg_loss, loss_val)
            push!(logg_grad, grad_val)
        end

        if t%solver.target_update_freq == 0
            run(graph.sess, graph.update_op)
        end

        if t%solver.eval_freq == 0
            push!(scores_eval, eval_q(graph, env, n_eval=solver.num_ep_eval,
                                                  max_episode_length=solver.max_episode_length))
        end

        if t%solver.log_freq == 0
            push!(logg_mean, avg100_reward)
            if solver.verbose
                logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3f | Grad %2.3f",
                                 t, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
                println(logg)
            end
        end
    end
    return logg_mean, logg_loss , logg_grad, episode_rewards, scores_eval
end

"""
Evaluate a Q network
"""
function eval_q(graph::TrainGraph,
                env::Union{MDPEnvironment, POMDPEnvironment};
                n_eval::Int64=100,
                max_episode_length::Int64=100)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= max_episode_length
            action =  get_action(graph, env, obs)
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
