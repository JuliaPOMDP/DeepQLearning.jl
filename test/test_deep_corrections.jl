using DeepQLearning, POMDPModels, DeepRL
using Base.Test

include("test_env.jl")

rng = MersenneTwister(1)
mdp = TestMDP((5,5), 4, 6)
dqn_solver = DeepQLearningSolver(max_steps=20000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = false, dueling=false, rng=rng)
solver = DeepCorrectionSolver(dqn_solver)

pol = solve(solver, mdp)


# value table 
using DiscreteValueIteration
mdp = GridWorld()
vi_pol = solve(ValueIterationSolver(), mdp)

function DeepQLearning.lowfi_values(mdp::GridWorld, s::Array{Float64})
    s_gw = convert_s(GridWorldState, s, mdp)
    si = state_index(mdp, s_gw)
    return vi_pol.qmat[si, :]
end

corr_pol = solve(solver, mdp)

