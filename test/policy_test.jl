using POMDPs, POMDPModels, Parameters
using Distributions, TensorFlow
const tf = TensorFlow
using DeepRL
using Gadfly


policy = restore()



env = POMDPEnvironment(TestPOMDP((5,5), 4, 6))


solver = DeepQLearningSolver(max_steps=10000, lr=0.001, eval_freq=1000,
                            arch = QNetworkArchitecture(conv=[], fc=[]))



train_graph = build_graph(solver, env)


# init and popuplate replay buffer
replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
populate_replay_buffer!(replay, env, max_pop=solver.train_start)

# init variables
run(train_graph.sess, global_variables_initializer())

logg_mean, logg_loss , logg_grad, episode_rewards, scores_eval = dqn_train(solver, env, train_graph, replay)

policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)


rng = MersenneTwister(1)

rtot = 0.
s = initial_state(env.problem, rng)
o = generate_o(env.problem, s, rng)
while !isterminal(env.problem, s)
    a = action(policy, o)
    sp, o, r = generate_sor(env.problem, s, a, rng)
    s = deepcopy(sp)
    rtot += r
end
rtot


save(solver, policy)



Gadfly.plot(x=1:length(logg_loss), y=logg_loss, Geom.line)


Gadfly.plot(x=1:length(logg_grad), y=logg_grad, Geom.line)

Gadfly.plot(x=1:length(logg_mean), y=logg_mean, Geom.line)

Gadfly.plot(x=1:length(episode_rewards), y=episode_rewards, Geom.line)


Gadfly.plot(x=1:length(scores_eval), y=scores_eval, Geom.line)
