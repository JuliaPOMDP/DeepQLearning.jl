using DeepQLearning
using POMDPModels
using POMDPSimulators
using Flux
using Random
using DeepRL
using Test

include("test_env.jl")
rng = MersenneTwister(1)

@testset "vanilla DQN" begin 
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8), Dense(8, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = false, dueling=false, prioritized_replay=false,
                                 rng=rng)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_basic = simulate(sim, mdp, policy)
    @test r_basic >= 1.5
end

@testset "double Q DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8), Dense(8, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork=model,max_steps=10000, learning_rate=0.005, eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = true, dueling=false, prioritized_replay=false,
                                 rng=rng)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_double = simulate(sim, mdp, policy)
    @test r_double >= 1.5
end

@testset "dueling DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8), Dense(8, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = false, dueling=true, prioritized_replay=false,
                                 rng=rng)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_duel = simulate(sim, mdp, policy)
    @test r_duel >= 1.5
end

@testset "Prioritized DDQN" begin 
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8), Dense(8, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = true, dueling=true, prioritized_replay=true,
                                 rng=rng)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_ddqn = simulate(sim, mdp, policy)
    @test r_ddqn >= 1.5
end

# DRQN tests 

@testset "TestMDP DRQN" begin
    mdp = TestMDP((5,5), 1, 6)
    model = Chain(x->flattenbatch(x), LSTM(25, 32), Dense(32, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500, 
                                 double_q = false, dueling=false, recurrence=true,
                                 rng=rng)
    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_drqn = simulate(sim, mdp, policy)
    @test r_drqn >= 1.5
end

@testset "GridWorld DDRQN" begin
    mdp = SimpleGridWorld();
    model = Chain(x->flattenbatch(x), LSTM(2, 32), Dense(32, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=10000, learning_rate=0.005,log_freq=500,
                             recurrence=true,trace_length=10, double_q=true, dueling=true)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=rng, max_steps=10)
    r_drqn = simulate(sim, mdp, policy)
    @test r_drqn >= 0.
end

@testset "BabyPOMDP DDRQN" begin 
    pomdp = TigerPOMDP(0.01, -1.0, 0.1, 0.8, 0.95);
    input_dims = reduce(*, size(convert_o(Vector{Float64}, first(observations(pomdp)), pomdp)))
    model = Chain(x->flattenbatch(x), LSTM(input_dims, 4), Dense(4, n_actions(pomdp)))
    solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=10000, learning_rate=0.0001,
                             log_freq=500, target_update_freq = 1000,
                             recurrence=true,trace_length=10, double_q=true, dueling=true, max_episode_length=100)

    policy = solve(solver, pomdp)
end

