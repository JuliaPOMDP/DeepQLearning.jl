using TensorFlow
using Random
graph1 = Graph()
policy1 = DeepQLearning.restore(problem_file = "log1/problem.bson", weights_file = "log1/weights.jld2", graph=graph1)

graph2 = Graph()
policy2 = DeepQLearning.restore(problem_file = "log2/problem.bson", weights_file = "log2/weights.jld2", graph=graph2)


mdp1 = TestMDP((5,5), 4, 6)
mdp2 = TestMDP((10,10), 4, 6)
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
