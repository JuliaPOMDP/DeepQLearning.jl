using Revise
using Random
using POMDPs
using POMDPModels
using DeepQLearning
using ProfileView
using Flux
using Profile
using BenchmarkTools


using Revise
using Random
using POMDPs
using DeepQLearning
using Flux
rng = MersenneTwister(1)
include("test/test_env.jl")
mdp = TestMDP((5,5), 4, 6)
model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, n_actions(mdp)))
solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                eval_freq=2000,num_ep_eval=100,
                                log_freq = 500,
                                double_q = false, dueling=true, prioritized_replay=false,
                                rng=rng)

policy = solve(solver, mdp)


mdp = SimpleGridWorld()

model = Chain(Dense(2, 32, relu), LSTM(32,32), Dense(32, 32, relu), Dense(32, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=1000, learning_rate=0.001,log_freq=500,
                             recurrence=true,trace_length=10, double_q=false, dueling=false, rng=rng, verbose=false)
policy = solve(solver, mdp)


@profile 1+1
Profile.clear()

@profile solve(solver, mdp)

ProfileView.view()



### get q_sa 
na = 4
bs = 32
tl = 100

s_batch = [rand(2, bs) for i=1:solver.trace_length]
a_batch = [rand(1:na, bs) for i=1:solver.trace_length]
q_values = model.(s_batch) # vector of size trace_length n_actions x batch_size
q_sa = [diag(view(q_values[i], a_batch[i], :)) for i=1:solver.trace_length]


using Flux: onehotbatch

onehotbatch(a_batch[1], 1:4)



a_batch[1]

sbatch = [rand(2, 32) for i=1:tl]
q_values = model.(sbatch)
qsa = getindex.(q_values, a_batch)

