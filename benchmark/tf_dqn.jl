using Random
using POMDPs
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
    solver = DeepQLearningSolver(arch=QNetworkArchitecture(conv=[], fc=[32,32]), 
                                 max_steps=10000, lr=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 15000, verbose=false,
                                 double_q = true, dueling=true, prioritized_replay=true,
                                 rng=rng)

    policy = solve(solver, mdp)
end

function bench_drqn(obsdim)
    rng = MersenneTwister(1)
    mdp = TestMDP(obsdim, 1, 6)
    solver = DeepRecurrentQLearningSolver(arch = RecurrentQNetworkArchitecture(fc_in=[], lstm_size=32,fc_out=[32]),
                                 max_steps=10000, lr=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 15000, verbose=false,
                                 double_q = true, dueling=false,
                                 rng=rng)
    policy = solve(solver, mdp)
end

for obsdim in [(5,5), (5,5,5), (20,20), (200,)]
    @show obsdim 
    @info "Prioritized DDQN"
    @btime bench_prioritized_ddqn($obsdim)
    @info "DRQN"
    @btime bench_drqn($obsdim)
end
