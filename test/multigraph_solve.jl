rng = MersenneTwister(1)
mdp1 = TestMDP((5,5), 4, 6)
solver = DeepQLearningSolver(max_steps=20000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = false, dueling=false, verbose=false, logdir="log1")
policy1 = solve(solver, mdp1)
DeepQLearning.save(solver, policy1, weights_file=solver.logdir*"/weights.jld2", problem_file=solver.logdir*"/problem.jld2")


mdp2 = TestMDP((10,10), 4, 6)
solver = DeepQLearningSolver(max_steps=20000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[8]),
                            save_freq = 2000, log_freq = 500,
                            double_q = false, dueling=false, verbose=false, logdir="log2")
policy2 = solve(solver, mdp2)
DeepQLearning.save(solver, policy2, weights_file=solver.logdir*"/weights.jld2", problem_file=solver.logdir*"/problem.jld2")

sim = RolloutSimulator(rng=MersenneTwister(0), max_steps=10)
r1 = simulate(sim, mdp1, policy1)
@test r1 > 1.5
println("reward ", r1)
println("placeholder ", policy1.s)

r2 = simulate(sim, mdp2, policy2)
@test r2 > 1.5
println("reward ", r2)
println("placeholder ", policy2.s)
println("placeholder policy1", policy1.s)
