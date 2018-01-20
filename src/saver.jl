function JLD.save(solver::DeepQLearningSolver,
                  policy::DQNPolicy;
                  weights_file::String = "weights.jld",
                  problem_file::String = "problem.jld")
    saver = tf.train.Saver()
    save(problem_file, "solver", solver, "env", policy.env)
    train.save(saver, policy.sess, weights_file)
end

function restore(;problem_file::String="problem.jld", weights_file::String="weights.jld")
    problem = load(problem_file)
    solver = problem[:solver],
    env = problem[:env]
    train_graph = build_graph(solver, env)
    saver = train.Saver()
    tf.train.restore(saver, train_graph.sess, weights_file)
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    return policy
end
