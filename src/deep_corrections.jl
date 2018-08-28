#=
Deprecated, moved to DeepCorrections.jl
=#

"""
Deep Correction algorithm to train an additive correction term to an existing value function 
""" 
mutable struct DeepCorrectionSolver <: Solver
    dqn::DeepQLearningSolver
end

struct DeepCorrectionPolicy <: AbstractNNPolicy
    q::Tensor # Q network
    s::Tensor # placeholder
    env::AbstractEnvironment
    sess
end

# the user should implement its own method
# should return a vector of size n_actions
function lowfi_values(problem, s) 
    na = n_actions(problem)
    return zeros(na)
end

function POMDPs.solve(solver::DeepCorrectionSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.dqn.rng)
    #init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepCorrectionSolver, env::AbstractEnvironment)
    train_graph = build_graph(solver.dqn, env)

    # init and populate replay buffer
    if solver.dqn.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.dqn.buffer_size, solver.dqn.batch_size)
    else
        replay = ReplayBuffer(env, solver.dqn.buffer_size, solver.dqn.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.dqn.train_start)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    # train model
    policy = DeepCorrectionPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    dqn_train(solver.dqn, env, train_graph, policy, replay)
    return policy
end

function get_action(policy::DeepCorrectionPolicy, o::Array{Float64})
    # cannot take a batch of observations
    q_low = lowfi_values(policy.env.problem, o)
    q_low = reshape(q_low, (1, length(q_low)))
    o_batch = reshape(o, (1, size(o)...))
    TensorFlow.set_def_graph(policy.sess.graph)
    q_corr = run(policy.sess, policy.q, Dict(policy.s => o_batch))
    q_val = q_low + q_corr
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end

