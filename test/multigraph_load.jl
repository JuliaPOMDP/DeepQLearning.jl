using DeepQLearning
using TensorFlow
include("test_env.jl")

graph1 = Graph()
policy1 = DeepQLearning.restore(problem_file = "log1/problem.jld", weights_file = "log1/weights.jld", graph=graph1)

graph2 = Graph()
policy2 = DeepQLearning.restore(problem_file = "log2/problem.jld", weights_file = "log2/weights.jld", graph=graph2)


mdp1 = TestMDP((5,5), 4, 6)
mdp2 = TestMDP((10,10), 4, 6)
sim = RolloutSimulator(rng=MersenneTwister(0), max_steps=10)

r1 = simulate(sim, mdp1, policy1)
println("reward ", r1)
println("placeholder ", policy1.s)

r2 = simulate(sim, mdp2, policy2)
println("reward ", r2)
println("placeholder ", policy2.s)
println("placeholder policy1", policy1.s)
