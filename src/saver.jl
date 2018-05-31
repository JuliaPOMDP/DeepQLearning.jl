function JLD.save(solver::Union{DeepQLearningSolver, DeepRecurrentQLearningSolver},
                  policy::Union{DQNPolicy, LSTMPolicy};
                  weights_file::String = "weights.jld",
                  problem_file::String = "problem.jld")
    saver = tf.train.Saver()
    warn("cannot save exploration policy")
    warn("cannot save evaluation function")
    solver_ = deepcopy(solver)
    solver_.exploration_policy = nothing
    solver_.evaluation_policy = nothing
    save(problem_file, "solver", solver_, "env", policy.env)
    train.save(saver, policy.sess, weights_file)
end

function restore(;problem_file::String="problem.jld", weights_file::String="weights.jld", graph=Graph())
    problem = load(problem_file)
    solver = problem["solver"]
    env = problem["env"]
    train_graph = build_graph(solver, env, graph)
    policy = restore_policy(env, solver, train_graph, weights_file)
    return policy
end


function restore_policy(env::AbstractEnvironment, solver::DeepQLearningSolver, train_graph::TrainGraph, weights_file::String)
    saver = train.Saver()
    tf.train.restore(saver, train_graph.sess, weights_file)
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
end

function restore_policy(env::AbstractEnvironment, solver::DeepRecurrentQLearningSolver, train_graph::RecurrentTrainGraph, weights_file::String)
    saver = train.Saver()
    tf.train.restore(saver, train_graph.sess, weights_file)
    policy = LSTMPolicy(train_graph.sess, env, solver.arch, solver.dueling)
end
