# using DeepQLearning
using Gadfly
using POMDPs, POMDPModels, Parameters, POMDPToolbox
using Distributions, TensorFlow
const tf = TensorFlow
using DeepRL
using DiscreteValueIteration

pomdp = TestPOMDP((5,5), 4, 6)
env = POMDPEnvironment(pomdp)
mdp = GridWorld()
env = MDPEnvironment(mdp)
sess= init_session()
solver = DeepQLearningSolver(max_steps=100000, lr=0.001, eval_freq=1000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[64,32]),
                            double_q = false, dueling=true)

policy, scores_eval, logg_mean, logg_loss, logg_grad, train_graph = solve(solver, mdp)



Gadfly.plot(x=1:length(logg_loss), y=logg_loss, Geom.line)


Gadfly.plot(x=1:length(logg_grad), y=logg_grad, Geom.line)




Gadfly.plot(x=1:length(logg_mean), y=logg_mean, Geom.line)

Gadfly.plot(x=1:length(episode_rewards), y=episode_rewards, Geom.line)



Gadfly.plot(x=1:length(scores_eval), y=scores_eval, Geom.line)



qvars = get_train_vars_by_name("active_q")

no_double_q_eval


maximum(scores_eval)


Gadfly.plot(x=1:length(no_double_q_eval), y=no_double_q_eval, Geom.line)



vi = ValueIterationSolver()
val_policy = solve(vi, mdp)

sim = RolloutSimulator(max_steps=30)
simulate(sim, mdp, val_policy)
simulate(sim, mdp, policy)

maximum(scores_eval)

policy

s

action(policy, s)

env = MDPEnvironment(mdp)

rng = MersenneTwister(1)

problem = mdp
rtot = 0.
step = 0
s = initial_state(problem, rng)
# o = generate_o(env.problem, s, rng)
acts = []
while !isterminal(problem, s) && step < 100
    a = action(policy, s)
    push!(acts, a)
    sp, r = generate_sr(problem, s, a, rng)
    s = deepcopy(sp)
    rtot += r
    step += 1
end
rtot


s
acts

save(solver, policy)
