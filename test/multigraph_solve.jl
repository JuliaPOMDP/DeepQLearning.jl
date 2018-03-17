using DeepQLearning

include("test_env.jl")

rng = MersenneTwister(1)
mdp = TestMDP((5,5), 4, 6)
solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=1000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            double_q = false, dueling=false, verbose=false, logdir="log1")
policy = solve(solver, mdp)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")


mdp = TestMDP((10,10), 4, 6)
solver = DeepQLearningSolver(max_steps=10000, lr=0.005, eval_freq=1000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            double_q = false, dueling=false, verbose=false, logdir="log2")
policy = solve(solver, mdp)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")
