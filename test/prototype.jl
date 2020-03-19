using Revise
using Random
using BenchmarkTools
using POMDPs
using POMDPModelTools
# using CuArrays
using Flux
using DeepQLearning
include("test/test_env.jl")

mdp = TestMDP((5,5), 4, 6)
# mdp = SimpleGridWorld()
rng = MersenneTwister(1)
mdp = TestMDP((5,5), 1, 6)
model = Chain(x-> flattenbatch(x), Dense(25, 32), Dense(32,32), Dense(32, 32, relu), Dense(32, length(actions(mdp))))

solver = DeepQLearningSolver(batch_size = 128, eval_freq = 10_000, save_freq=10_000, qnetwork = model, prioritized_replay=true, 
                             max_steps=1000, learning_rate=0.001,train_start=500,log_freq=100000, 
                             recurrence=false,trace_length=10, double_q=true, dueling=false, rng=rng, verbose=false)

@btime policy = solve($solver, $mdp)


policy = solve(solver, mdp)

env = MDPEnvironment(mdp)
o = reset!(env)

using RLInterface
using LinearAlgebra

env = MDPEnvironment(mdp)
replay = DeepQLearning.initialize_replay_buffer(solver, env)
active_q = solver.qnetwork
policy = NNPolicy(env.problem, active_q, ordered_actions(env.problem), length(obs_dimensions(env)))
active_q = getnetwork(policy) 
target_q = deepcopy(active_q)


s_batch, a_batch, r_batch, sp_batch, done_batch, indices, importance_weights = DeepQLearning.sample(replay)

q_values = active_q(s_batch)
best_a = Flux.onecold(q_values)



argmax(q_values, dims=1)

best_a = [CartesianIndex(Flux.argmax(q_values[:, i]), i) for i=1:solver.batch_size]
q_values[best_a] == dropdims(maximum(q_values, dims=1), dims=1)

a_batch
q_values[a_batch]

@btime q_sa = diag(view($q_values, $a_batch, :));
@btime q_sa = q_values[a, :]
q_sa2 = q_values[:][a_batch]

@btime getindex.(Ref($q_values), $a_batch, 1:$solver.batch_size);
a_batch = CartesianIndex.(a_batch, 1:solver.batch_size)

A = rand(4, 32)
a = rand(1:4, 32) # indices of the first column of A 
@btime diag(view($A, $a, :)); # first way 
@btime getindex.(Ref($A), $a, 1:size($A, 2));

function test(active_q, target_q)
    p = Flux.params(active_q)
    loss_val = nothing
    td_err = nothing
    gs = Flux.gradient(p) do 
        # compute q_vals and targets 
        q_values = active_q(s_batch)
        q_sa = diag(view(q_values, a_batch, :))
        γ = discount(env.problem)
        q_sp_max = dropdims(maximum(target_q(sp_batch), dims=1), dims=1)
        q_targets = r_batch .+ (1f0 .- done_batch) .* γ .* q_sp_max
        td_err = q_sa .- q_targets
        loss_val = mean(huber_loss, importance_weights.*q_sa)
        loss_val
    end
    # for (i, g) in gs.grads
    #     @show g
    #     @show g.contents
    # end
    DeepQLearning.globalnorm(p, gs)
    return loss_val, td_err
end

gs.grads

@show [a for (i, a) in gs.grads]






# benchmark
# standard dqn 650ms 282MB
# double dqn   706ms 329MB
# zygote dqn   764.5 175MB
# ddqn zygote  832.5 184MB

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
