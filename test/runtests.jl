using DeepQLearning, POMDPModels, DeepRL
using Base.Test

include("tf_helpers_test.jl")
include("test_env.jl")

rng = MersenneTwister(1)
mdp = TestMDP((5,5), 4, 6)
solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = false, dueling=false, rng=rng)

policy = solve(solver, mdp)
sim = RolloutSimulator(rng=rng, max_steps=10)
r_basic = simulate(sim, mdp, policy)
@test r_basic >= 1.5
solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000,
                            log_freq = 500,
                            double_q = true, dueling=false, rng=rng)
policy_double_q = solve(solver, mdp)
r_double_q = simulate(sim, mdp, policy_double_q)
@test r_double_q >= 1.5

solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = false, dueling=true, rng=rng)
policy_dueling = solve(solver, mdp)
r_dueling = simulate(sim, mdp, policy_dueling)
@test r_dueling >= 1.5

solver = DeepRecurrentQLearningSolver(max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = RecurrentQNetworkArchitecture(fc_in=[8], lstm_size=12, fc_out=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = true, dueling=true, rng=rng)

mdp = TestMDP((5,5), 1, 6)
policy = solve(solver, mdp)
avg_test = eval_lstm(policy, MDPEnvironment(mdp), policy.sess)
@test avg_test >= 1.5

solver = DeepRecurrentQLearningSolver(max_steps=20000, lr=0.005, eval_freq=2000, num_ep_eval=100,
                            arch = RecurrentQNetworkArchitecture(fc_in=[8], lstm_size=12, fc_out=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q=false, dueling=false, grad_clip=false, rng=rng)
mdp = GridWorld();
policy = solve(solver, mdp)
avg_gridworld = eval_lstm(policy, MDPEnvironment(mdp), policy.sess)
@test avg_gridworld > 1.5

include("multigraph_solve.jl")
include("multigraph_load.jl")