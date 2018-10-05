
function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment)
    train_graph = build_graph(solver, env)

    replay = initialize_replay_buffer(solver, env)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    # train model
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    dqn_train(solver, env, train_graph, policy, replay)
    return policy
end

function initialize_replay_buffer(solver::DeepQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end


function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    #init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end

function dqn_train(solver::DeepQLearningSolver,
                   env::AbstractEnvironment,
                   graph::G,
                   policy::AbstractNNPolicy,
                   replay::Union{ReplayBuffer, PrioritizedReplayBuffer}) where G
    summary_writer = tf.summary.FileWriter(solver.logdir)
    reset_hidden_state!(policy)
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
            loss_val, td_errors, grad_val = batch_train!(env, graph, replay)
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
            logger(solver, summary_writer,
                avg100_reward, scores_eval, loss_val, td_errors, grad_val, eps, avg100_steps, episode_rewards, episode_steps, t)
        end

        if t > solver.train_start && t%solver.save_freq == 0
            model_saved, saved_mean_reward = save_model(solver, graph, scores_eval, saved_mean_reward, model_saved)
        end

    end
    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_mean_reward)
            saver = tf.train.Saver()
            train.restore(saver, graph.sess, solver.logdir*"/weights.jld")
        end
    end
    return
end

function batch_train!(env::AbstractEnvironment, graph::TrainGraph, replay::ReplayBuffer)
    weights = ones(replay.batch_size)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    return batch_train!(graph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights)
end

function batch_train!(env::AbstractEnvironment, graph::TrainGraph, replay::PrioritizedReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    loss_val, td_errors, grad_val = batch_train!(graph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights)
    update_priorities!(replay, indices, td_errors)
    return loss_val, td_errors, grad_val
end

function batch_train!(graph::TrainGraph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights)
    tf.set_def_graph(graph.sess.graph)
    feed_dict = Dict(graph.s => s_batch,
                    graph.a => a_batch,
                    graph.sp => sp_batch,
                    graph.r => r_batch,
                    graph.done_mask => done_batch,
                    graph.importance_weights => weights)
    loss_val, td_errors, grad_val, _ = run(graph.sess,[graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                feed_dict)
    return (loss_val, td_errors, grad_val)
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
    replay = initialize_replay_buffer(solver, env)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    policy = train_graph.lstm_policy
    policy.sess = train_graph.sess
    # train model
    drqn_train(solver, env, train_graph, policy, replay)
    return policy
end

function initialize_replay_buffer(solver::DeepRecurrentQLearningSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end


function drqn_train(solver::DeepRecurrentQLearningSolver,
                   env::AbstractEnvironment,
                   graph::RecurrentTrainGraph,
                   policy::LSTMPolicy,
                   replay::EpisodeReplayBuffer)
    summary_writer = tf.summary.FileWriter(solver.logdir)
    obs = reset(env)
    reset_hidden_state!(policy)
    done = false
    step = 0
    rtot = 0
    episode = DQExperience[]
    sizehint!(episode, solver.max_episode_length)
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    saved_mean_reward = 0.
    model_saved = false
    scores_eval = 0.0
    eps = 1.0
    weights = ones(solver.batch_size*solver.trace_length)
    init_c = zeros(solver.batch_size, solver.arch.lstm_size)
    init_h = zeros(solver.batch_size, solver.arch.lstm_size)
    grad_val, loss_val = -1, -1 # sentinel value
    for t=1:solver.max_steps
        act, eps = exploration(solver.exploration_policy, policy, env, obs, t, solver.rng)
        ai = actionindex(env.problem, act)
        op, rew, done, info = step!(env, act)
        exp = DQExperience(obs, ai, rew, op, done)
        push!(episode, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            add_episode!(replay, episode)
            episode = DQExperience[] # empty episode
            obs = reset(env)
            reset_hidden_state!(policy)
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
            loss_val, td_errors, grad_val, _ = run(graph.sess,
                                                       [graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                                       feed_dict)
        end

        if t%solver.target_update_freq == 0
            run(graph.sess, graph.update_op)
        end

        if t%solver.eval_freq == 0
            # save hidden state before
            hidden_state = deepcopy(graph.lstm_policy.state_val)
            scores_eval = evaluation(solver.evaluation_policy, 
                                 policy, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
            # reset hidden state
            graph.lstm_policy.state_val = hidden_state
        end
        if t%solver.log_freq == 0
            # log to tensorboard
            logger(solver, summary_writer,
                avg100_reward, scores_eval, loss_val, td_errors, grad_val, eps, avg100_steps, episode_rewards, episode_steps, t)
        end

        if t > solver.train_start && t%solver.save_freq == 0
            model_saved, saved_mean_reward = save_model(solver, graph, scores_eval, saved_mean_reward, model_saved)
        end
    end
    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_mean_reward)
            saver = tf.train.Saver()
            train.restore(saver, graph.sess, solver.logdir*"/weights.jld")
        end
    end
    return
end

function logger(solver::Union{DeepQLearningSolver, DeepRecurrentQLearningSolver}, summary_writer, 
                avg100_reward, scores_eval, loss_val, td_errors, grad_val, eps, avg100_steps, episode_rewards, episode_steps, t)
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
    if  solver.verbose
        logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e",
                            t, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
        println(logg)
    end        
end

function save_model(solver::Union{DeepQLearningSolver, DeepRecurrentQLearningSolver}, graph,
                    scores_eval, saved_mean_reward, model_saved, weights_file::String=solver.logdir*"/weights.jld")
    if scores_eval[end] >= saved_mean_reward
        if solver.verbose
            println("Saving new model with eval reward ", scores_eval[end])
        end
        saver = tf.train.Saver()
        train.save(saver, graph.sess, weights_file)
        model_saved = true
        saved_mean_reward = scores_eval[end]
    end
    return model_saved, saved_mean_reward
end
