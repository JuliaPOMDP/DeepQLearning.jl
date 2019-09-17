using Revise
using Random
using POMDPs
using POMDPModels
using DeepQLearning
using Flux
using FileIO
using JLD2

rng = MersenneTwister(1)
mdp = SimpleGridWorld()
model = Chain(x->flattenbatch(x), Dense(2, 32, tanh), Dense(32, length(actions(mdp))))
solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                eval_freq=2000,num_ep_eval=100,
                                log_freq = 500,
                                double_q = true, dueling=false, prioritized_replay=true,verbose=true,
                                rng=rng)

policy = solve(solver, mdp)

@save "policy.jld2" policy 

using StaticArrays

@load "policy.jld2" policy

using Revise
using Random
using BenchmarkTools
using POMDPs
using POMDPModelTools
# using CuArrays
using Flux
using DeepQLearning
include("test/test_env.jl")
# mdp = TestMDP((5,5), 4, 6)
# mdp = SimpleGridWorld()
rng = MersenneTwister(1)
mdp = TestMDP((5,5), 1, 6)
model = Chain(x-> flattenbatch(x), Dense(25, 32), Dense(32,32), Dense(32, 32, relu), Dense(32, length(actions(mdp))))

solver = DeepQLearningSolver(batch_size = 128, eval_freq = 10_000, save_freq=10_000, qnetwork = model, prioritized_replay=true, max_steps=1000, learning_rate=0.001,log_freq=5000,
                             recurrence=false,trace_length=10, double_q=false, dueling=false, rng=rng, verbose=false)

@btime policy = solve($solver, $mdp)

using Profile
using ProfileView
@profile 1+1
Profile.clear()

@profile solve(solver, mdp)

ProfileView.view()

### Try on SubHunt

using Revise
using POMDPs
using SubHunt
using RLInterface
using DeepQLearning
using Flux
 
solver = DeepQLearningSolver(qnetwork= Chain(Dense(8, 32, relu), Dense(32,32,relu), Dense(32, 6)), 
                             max_steps=100_000)
solve(solver, SubHuntPOMDP())



### get q_sa 
na = 4
bs = 32
tl = 100

s_batch = [rand(2, bs) for i=1:solver.trace_length]
a_batch = [rand(1:na, bs) for i=1:solver.trace_length]
q_values = model.(s_batch) # vector of size trace_length nactions x batch_size
q_sa = [diag(view(q_values[i], a_batch[i], :)) for i=1:solver.trace_length]


using Flux: onehotbatch

onehotbatch(a_batch[1], 1:4)



a_batch[1]

sbatch = [rand(2, 32) for i=1:tl]
q_values = model.(sbatch)
qsa = getindex.(q_values, a_batch)

using POMDPModels
using RLInterface

env = POMDPEnvironment(TigerPOMDP())

function simulate(env::AbstractEnvironment, nsteps::Int = 10)
    done = false
    r_tot = 0.0
    step = 1
    o = reset!(env)
    while !done && step <= nsteps
        action = sample_action(env) # take random action 
        obs, rew, done, info = step!(env, action)
        @show obs, rew, done, info
        r_tot += rew
        step += 1
    end
    return r_tot
end

@show simulate(env)