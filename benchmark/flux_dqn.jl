using POMDPs
using Random
using Flux
using POMDPSimulators
using DeepQLearning
using BenchmarkTools

include("../test/test_env.jl")

function evaluate(mdp, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end

function bench_prioritized_ddqn(obsdim)
    rng = MersenneTwister(1)
    mdp = TestMDP(obsdim, 4, 6)
    model = Chain(x->flattenbatch(x), Dense(reduce(*, obsdim)*4, 32), Dense(32, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 15000, verbose=false,
                                 double_q = true, dueling=true, prioritized_replay=true,
                                 rng=rng)

    policy = solve(solver, mdp)
    r_ddqn =  evaluate(mdp, policy, rng)
end

function bench_drqn(obsdim)
    rng = MersenneTwister(1)
    mdp = TestMDP(obsdim, 1, 6)
    model = Chain(x->flattenbatch(x), LSTM(reduce(*, obsdim), 32), Dense(32, n_actions(mdp)))
    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,trace_length=10,
                                 log_freq = 15000, verbose=false,
                                 double_q = true, dueling=false, recurrence=true,
                                 rng=rng)
    policy = solve(solver, mdp)
    r_drqn =  evaluate(mdp, policy, rng)
end

for obsdim in [(5,5), (5,5,5), (20,20), (200,)]
    @show obsdim 
    @info "Prioritized DDQN"
    @btime bench_prioritized_ddqn($obsdim)
    @info "DRQN"
    @btime bench_drqn($obsdim)
end