
function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment)
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
    # train model
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    dqn_train(solver, env, train_graph, policy, replay)
    return policy
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    #init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end

function dqn_train(solver::DeepQLearningSolver,
                   env::AbstractEnvironment,
                   graph::TrainGraph,
                   policy::AbstractNNPolicy,
                   replay::Union{ReplayBuffer, PrioritizedReplayBuffer})
    summary_writer = tf.summary.FileWriter(solver.logdir)
    obs = reset(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    saved_mean_reward = 0.
    scores_eval = 0.
    eps = 1.0
    weights = ones(solver.batch_size)
    model_saved = false
    for t=1:solver.max_steps
        act, eps = exploration(solver.exploration_policy, policy, env, obs, t, solver.rng)
        ai = action_index(env.problem, act)
        op, rew, done, info = step!(env, act)
        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            obs = reset(env)
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

        end

        if t%solver.target_update_freq == 0
            run(graph.sess, graph.update_op)
        end

        if t%solver.eval_freq == 0
            scores_eval = evaluation(solver.evaluation_policy, 
                                 policy, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
        end

        if t%solver.log_freq == 0
            # log to tensorboard
            tb_avgr = logg_scalar(avg100_reward, "avg_reward")
            tb_evalr = logg_scalar(scores_eval[end], "eval_reward")
            tb_loss = logg_scalar(loss_val, "loss")
            tb_tderr = logg_scalar(mean(td_errors), "mean_td_error")
            tb_grad = logg_scalar(grad_val, "grad_norm")
            tb_eps = logg_scalar(eps, "epsilon")
            tb_avgs = logg_scalar(avg100_steps, "avg_steps")
            if length(episode_rewards) > 1
                tb_epreward = logg_scalar(episode_rewards[end-1], "episode_reward")
                write(summary_writer, tb_epreward, t)
            end
            if length(episode_steps) >= 1
                tb_epstep = logg_scalar(episode_steps[end], "episode_steps")
                write(summary_writer, tb_epstep, t)
            end
            write(summary_writer, tb_avgr, t)
            write(summary_writer, tb_evalr, t)
            write(summary_writer, tb_loss, t)
            write(summary_writer, tb_tderr, t)
            write(summary_writer, tb_grad, t)
            write(summary_writer, tb_eps, t)
            write(summary_writer, tb_avgs, t)
            if solver.verbose
                logg = @sprintf("%5d / %5d eps %0.3e |  avgR %1.3e | Loss %2.3e | Grad %2.3e",
                                 t, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
                println(logg)
            end
        end

        if t > solver.train_start && t%solver.save_freq == 0
            if scores_eval[end] >= saved_mean_reward
                if solver.verbose
                    println("Saving new model with eval reward ", scores_eval[end])
                end
                saver = tf.train.Saver()
                train.save(saver, graph.sess, solver.logdir*"weights.jld")
                model_saved = true
                saved_mean_reward = scores_eval[end]
            end
        end

    end
    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_mean_reward)
            saver = tf.train.Saver()
            train.restore(saver, graph.sess, solver.logdir*"weights.jld")
        end
    end
    return
end


function POMDPs.solve(solver::DeepRecurrentQLearningSolver, problem::Union{MDP,POMDP})
    if !isa(problem, POMDP)
        env = MDPEnvironment(problem, rng=solver.rng)
    else
        env = POMDPEnvironment(problem, rng=solver.rng)
    end
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepRecurrentQLearningSolver, env::AbstractEnvironment)
    #init session and build graph Create a TrainGraph object with all the tensors
    train_graph = build_graph(solver, env)
    # init and populate replay buffer
    replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    # train model
    drqn_train(solver, env, train_graph, replay)
    policy = train_graph.lstm_policy
    policy.sess = train_graph.sess
    return policy
end


function drqn_train(solver::DeepRecurrentQLearningSolver,
                   env::AbstractEnvironment,
                   graph::RecurrentTrainGraph,
                   replay::EpisodeReplayBuffer)
    summary_writer = tf.summary.FileWriter(solver.logdir)
    obs = reset(env)
    reset_hidden_state!(graph.lstm_policy)
    done = false
    step = 0
    rtot = 0
    episode = DQExperience[]
    sizehint!(episode, solver.max_episode_length)
    episode_rewards = Float64[0.0]
    saved_mean_reward = 0.
    model_saved = false
    scores_eval = 0.0
    eps = 1.0
    weights = ones(solver.batch_size*solver.trace_length)
    init_c = zeros(solver.batch_size, solver.arch.lstm_size)
    init_h = zeros(solver.batch_size, solver.arch.lstm_size)
    grad_val, loss_val = -1, -1 # sentinel value
    for t=1:solver.max_steps
        if rand(solver.rng) > eps
            action = get_action!(graph.lstm_policy, obs, graph.sess)
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
        push!(episode, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            add_episode!(replay, episode)
            episode = DQExperience[] # empty episode
            obs = reset(env)
            reset_hidden_state!(graph.lstm_policy)
            push!(episode_rewards, 0.0)
            done = false
            step = 0
            rtot = 0
        end
        num_episodes = length(episode_rewards)
        avg100_reward = mean(episode_rewards[max(1, length(episode_rewards)-101):end])
        if t%solver.train_freq == 0
            s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
            feed_dict = Dict(graph.s => s_batch,
                             graph.a => a_batch,
                             graph.sp => sp_batch,
                             graph.r => r_batch,
                             graph.done_mask => done_batch,
                             graph.trace_mask => trace_mask_batch,
                             graph.importance_weights => weights,
                             graph.hq_in.c => init_c,
                             graph.hq_in.h => init_h,
                             graph.hqp_in.c => init_c,
                             graph.hqp_in.h => init_h,
                             graph.target_hq_in.c => init_c,
                             graph.target_hq_in.h => init_h
                             )
            loss_val, td_errors_val, grad_val, _ = run(graph.sess,
                                                       [graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                                       feed_dict)
        end

        if t%solver.target_update_freq == 0
            run(graph.sess, graph.update_op)
        end

        if t%solver.eval_freq == 0
            # save hidden state before
            hidden_state = graph.lstm_policy.state_val
            scores_eval = eval_lstm(graph.lstm_policy,
                                     env,
                                     graph.sess,
                                     n_eval=solver.num_ep_eval,
                                     max_episode_length=solver.max_episode_length,
                                     verbose = solver.verbose)
            # reset hidden state
            graph.lstm_policy.state_val = hidden_state
        end

        if t%solver.log_freq == 0
            # log to tensorboard
            tb_avgr = logg_scalar(avg100_reward, "avg_reward")
            tb_evalr = logg_scalar(scores_eval[end], "eval_reward")
            tb_loss = logg_scalar(loss_val, "loss")
            tb_tderr = logg_scalar(mean(td_errors_val), "mean_td_error")
            tb_grad = logg_scalar(grad_val, "grad_norm")
            tb_epreward = logg_scalar(episode_rewards[end], "episode_reward")
            tb_eps = logg_scalar(eps, "epsilon")
            write(summary_writer, tb_avgr, t)
            write(summary_writer, tb_evalr, t)
            write(summary_writer, tb_loss, t)
            write(summary_writer, tb_tderr, t)
            write(summary_writer, tb_grad, t)
            write(summary_writer, tb_epreward, t)
            write(summary_writer, tb_eps, t)
            if  solver.verbose
                logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3f | Grad %2.3f",
                                 t, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
                println(logg)
            end
        end
        if t > solver.train_start && t%solver.save_freq == 0
            if scores_eval[end] >= saved_mean_reward
                if solver.verbose
                    println("Saving new model with eval reward ", scores_eval[end])
                end
                saver = tf.train.Saver()
                train.save(saver, graph.sess, solver.logdir*"weights.jld")
                model_saved = true
                saved_mean_reward = scores_eval[end]
            end
        end
    end
    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_mean_reward)
        end
    end
    return
end

function eval_lstm(policy::LSTMPolicy,
                env::AbstractEnvironment,
                sess;
                n_eval::Int64=100,
                max_episode_length::Int64=100,
                verbose::Bool=false)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset_hidden_state!(policy)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= max_episode_length
            action = get_action!(policy, obs, sess)
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
    if verbose
        println("Evaluation ... Avg Reward ", avg_r/n_eval)
    end
    return  avg_r /= n_eval
end
