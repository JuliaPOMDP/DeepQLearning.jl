using DeepQLearning
using POMDPModels
using POMDPSimulators
using Random
using DeepRL
using Test

include("tf_helpers_test.jl")
include("test_env.jl")


rng = MersenneTwister(1)

@testset "vanilla DQN" begin 
    mdp = TestMDP((5,5), 4, 6)
    solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                                arch = QNetworkArchitecture(conv=[], fc=[8]),
                                save_freq = 2000, log_freq = 500,
                                double_q = false, dueling=false, rng=rng)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_basic = simulate(sim, mdp, policy)
    @test r_basic >= 1.5
end

@testset "double Q DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                                arch = QNetworkArchitecture(conv=[], fc=[8]),
                                save_freq = 2000,
                                log_freq = 500,
                                double_q = true, dueling=false, rng=rng)
    policy_double_q = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_double_q = simulate(sim, mdp, policy_double_q)
    @test r_double_q >= 1.5
end

@testset "dueling DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                                arch = QNetworkArchitecture(conv=[], fc=[8]),
                                save_freq = 2000, log_freq = 500,
                                double_q = false, dueling=true, rng=rng)
    policy_dueling = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_dueling = simulate(sim, mdp, policy_dueling)
    @test r_dueling >= 1.5
end

# DRQN tests 

@testset "TestMDP DRQN" begin
    n_eval = 100
    max_steps = 100
    solver = DeepRecurrentQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                                arch = RecurrentQNetworkArchitecture(fc_in=[8], lstm_size=12, fc_out=[8]),
                                save_freq = 2000, log_freq = 500,
                                double_q = true, dueling=true, rng=rng)

    mdp = TestMDP((5,5), 1, 6)
    policy = solve(solver, mdp)
    avg_test = basic_evaluation(policy, MDPEnvironment(mdp), n_eval, max_steps, false)
    @test avg_test >= 1.5
end

@testset "GridWorld DRQN" begin
    n_eval = 100
    max_steps = 100
    solver = DeepRecurrentQLearningSolver(max_steps=20000, lr=0.005, eval_freq=2000, num_ep_eval=100,
                                arch = RecurrentQNetworkArchitecture(fc_in=[8], lstm_size=12, fc_out=[8]),
                                save_freq = 2000, log_freq = 500,
                                double_q=false, dueling=false, grad_clip=false, rng=rng)
    mdp = SimpleGridWorld();
    policy = solve(solver, mdp)
    avg_gridworld = basic_evaluation(policy, MDPEnvironment(mdp), n_eval, max_steps, false)
    @test avg_gridworld > 1.5
end

@testset "multiple graphs" begin
    include("multigraph_solve.jl")
    include("multigraph_load.jl")
end
