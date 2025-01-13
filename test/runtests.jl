using DeepQLearning
using POMDPModels
using POMDPTools
using Flux
using Random
using StaticArrays
using Test
import CommonRLInterface

RL = CommonRLInterface

Random.seed!(7)
GLOBAL_RNG = MersenneTwister(1) # for test consistency

include("test_env.jl")

function evaluate(mdp::Union{MDP,POMDP}, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        DeepQLearning.resetstate!(policy)
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end

function evaluate(env::RL.AbstractEnv, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    for i=1:n_ep
        DeepQLearning.resetstate!(policy)
        r = 0.0
        step = 0
        RL.reset!(env)
        while !RL.terminated(env) && step < max_steps
            a = action(policy, RL.observe(env))
            r += RL.act!(env, a)
            step += 1
        end
        avg_r += r
    end
    return avg_r/=n_ep
end


@testset "vanilla DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, max_steps=max_steps, learning_rate=0.005,
                                 exploration_policy = exploration,
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = false, dueling=false, prioritized_replay=false)

    policy = solve(solver, mdp)
    r_basic = evaluate(mdp, policy, GLOBAL_RNG)
    @test r_basic >= 1.5
    @test size(actionvalues(policy, rand(GLOBAL_RNG, initialstate(mdp)))) == (length(actions(mdp)),)
end

@testset "double Q DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork=model,max_steps=max_steps, learning_rate=0.005, eval_freq=2000,num_ep_eval=100,
                                 exploration_policy = exploration,
                                 log_freq = 500,
                                 double_q = true, dueling=false, prioritized_replay=false)

    policy = solve(solver, mdp)
    r_double =  evaluate(mdp, policy, GLOBAL_RNG)
    @test r_double >= 1.5
end

@testset "dueling DQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, max_steps=max_steps, learning_rate=0.005,
                                 eval_freq=2000,num_ep_eval=100,
                                 exploration_policy = exploration,
                                 log_freq = 500,
                                 double_q = false, dueling=true, prioritized_replay=false)

    policy = solve(solver, mdp)
    r_duel =  evaluate(mdp, policy, GLOBAL_RNG)
    @test r_duel >= 1.5
end

@testset "Prioritized DDQN" begin
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, max_steps=max_steps, learning_rate=0.005,
                                 exploration_policy = exploration,
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500,
                                 double_q = true, dueling=true, prioritized_replay=true)

    policy = solve(solver, mdp)
    r_ddqn =  evaluate(mdp, policy, GLOBAL_RNG)
    @test r_ddqn >= 1.5
end

# DRQN tests

@testset "TestMDP DRQN" begin
    mdp = TestMDP((5,5), 1, 6)
    model = Chain(x->flattenbatch(x), LSTM(25, 8), Dense(8, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, max_steps=max_steps, learning_rate=0.005,
                                 exploration_policy = exploration,
                                 eval_freq=2000,num_ep_eval=100, 
                                 log_freq = 500,
                                 double_q = true, dueling=false, recurrence=true)
    policy = solve(solver, mdp)
    r_drqn =  evaluate(mdp, policy, GLOBAL_RNG)
    @test r_drqn >= 0.
end

@testset "GridWorld DDRQN" begin
    mdp = SimpleGridWorld();
    model = Chain(x->flattenbatch(x), LSTM(2, 32), Dense(32, length(actions(mdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=max_steps,
                                 exploration_policy = exploration, 
                                 learning_rate=0.001,log_freq=500,
                                 recurrence=true,trace_length=10, 
                                 double_q=true, dueling=true)

    policy = solve(solver, mdp)
    sim = RolloutSimulator(rng=GLOBAL_RNG, max_steps=10)
    r_drqn =  evaluate(mdp, policy, GLOBAL_RNG)
    @test r_drqn >= 0.
end

@testset "TigerPOMDP DDRQN" begin
    pomdp = TigerPOMDP(0.01, -1.0, 0.1, 0.8, 0.95);
    input_dims = reduce(*, size(convert_o(Vector{Float64}, first(observations(pomdp)), pomdp)))
    model = Chain(x->flattenbatch(x), LSTM(input_dims, 4), Dense(4, length(actions(pomdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(pomdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=max_steps,
                             learning_rate=0.0001, exploration_policy = exploration,
                             log_freq=500, target_update_freq = 1000,
                             recurrence=true,trace_length=10, double_q=true, dueling=true, max_episode_length=100)

    policy = solve(solver, pomdp)
    @test size(actionvalues(policy, true)) == (length(actions(pomdp)),)
end

@testset "Static Array Env" begin
    mutable struct StaticArrayMDP <: MDP{typeof(SVector(1)), Int64}
        state::typeof(SVector(1))
    end
    POMDPs.discount(::StaticArrayMDP) = 0.95f0
    function POMDPs.initialstate(m::StaticArrayMDP)
        ImplicitDistribution() do rng 
            m.state
        end
    end 
    
    function POMDPs.gen(m::StaticArrayMDP, s, a, rng::AbstractRNG)
        return (sp=s + SVector(a), r=m.state[1]^2)
    end
    
    POMDPs.isterminal(::StaticArrayMDP, s) = s[1] >= 3
    POMDPs.actions(::StaticArrayMDP) = [0,1]

    mdp = StaticArrayMDP(SVector(1))

    model = Chain(Dense(1, 32), Dense(32, length(actions(mdp))))

    max_steps = 10000
    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=5),
                                  rng=GLOBAL_RNG)

    solver = DeepQLearningSolver(qnetwork = model, max_steps=10, exploration_policy=exploration,
                                learning_rate=0.005,log_freq=500,
                                recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
    policy = solve(solver, mdp)

    @test evaluate(mdp, policy, GLOBAL_RNG) > 1.0
end

@testset "Common RL Env" begin
    mutable struct SimpleEnv <: RL.AbstractEnv
        s::Int
    end

    RL.reset!(env::SimpleEnv) = env.s = 1
    RL.state(env::SimpleEnv) = env.s
    RL.setstate!(env::SimpleEnv, s::Int) = env.s = s
    RL.setstate!(env::SimpleEnv, s::Float32) = env.s = Int(s)
    RL.setstate!(env::SimpleEnv, s::Vector{Float32}) = env.s = Int(s[1])
    RL.actions(env::SimpleEnv) = [-1, 1]
    RL.observe(env::SimpleEnv) = Float32[env.s]
    RL.terminated(env::SimpleEnv) = env.s >= 3
    function RL.act!(env::SimpleEnv, a)
        r = env.s
        env.s = max(1, env.s + a)
        return r
    end

    env = SimpleEnv(1)

    model = Chain(Dense(1, 32), Dense(32, length(RL.actions(env))))

    max_steps = 10000
    exploration = EpsGreedyPolicy(convert(MDP, env), LinearDecaySchedule(start=1.0, stop=0.01, steps=5),
                                  rng=GLOBAL_RNG)

    solver = DeepQLearningSolver(qnetwork = model, max_steps=10, exploration_policy=exploration,
                                learning_rate=0.005,log_freq=500,
                                recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
    policy = solve(solver, env)

    println("here")

    @test evaluate(env, policy, GLOBAL_RNG) > 1.0
end


@testset "README Examples" begin
    include("README_examples.jl")
end
