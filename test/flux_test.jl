using DeepQLearning
using POMDPModels
using DeepRL
using Test

include("test_env.jl")

mdp = TestMDP((5,5), 4, 6)

env = MDPEnvironment(mdp)
input_dims = obs_dimensions(env)
flattened_input_dims = reduce(*, input_dims)
output_dims = n_actions(env)
model = Chain(x->flattenbatch(x), Dense(flattened_input_dims, 8), Dense(8, output_dims))
# model = Chain(x->flattenbatch(x), LSTM(flattened_input_dims, 32), Dense(32, output_dims))
solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=true, max_steps=10000, learning_rate=0.005,log_freq=500,
                             recurrence=false,trace_length=6, double_q=false, dueling=true)

@time solve(solver, env)

### Hyperparameters
buffer_size = 3000
batch_size = 32
train_start = 200
learning_rate = 1e-4
trace_length = 6

env = MDPEnvironment(mdp)

### MODEL BUILDING 
input_dims = obs_dimensions(env)
flattened_input_dims = reduce(*, input_dims)
output_dims = n_actions(env)
model = Chain(x->flattenbatch(x), LSTM(flattened_input_dims, 32), Dense(32, output_dims))

active_q = model
target_q = deepcopy(active_q)

### Replay Buffer (unchanged from previous implementation)
replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
DeepQLearning.populate_replay_buffer!(replay, env, max_pop=solver.train_start)


### Batch Train, in Flux, the batch size is the last dimension
s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = DeepQLearning.sample(replay)

s_batch = batch_trajectories(s_batch, solver.trace_length, solver.batch_size)
a_batch = batch_trajectories(a_batch, solver.trace_length, solver.batch_size)
r_batch = batch_trajectories(r_batch, solver.trace_length, solver.batch_size)
sp_batch = batch_trajectories(sp_batch, solver.trace_length, solver.batch_size)
done_batch = batch_trajectories(done_batch, solver.trace_length, solver.batch_size)
trace_mask_batch = batch_trajectories(trace_mask_batch, solver.trace_length, solver.batch_size)

q_values = active_q.(s_batch) # vector of size trace_length n_actions x batch_size
q_sa = [zeros(eltype(q_values[1]), solver.batch_size) for i=1:solver.trace_length]
for i=1:solver.trace_length  # there might be a more elegant way of doing this
    for j=1:solver.batch_size
        if a_batch[i][j] != 0
            q_sa[i][j] = q_values[i][a_batch[i][j], j]
        end
    end
end
q_sp_max = vec.(maximum.(target_q.(sp_batch), dims=1))
q_targets = Vector{eltype(q_sa)}(undef, solver.trace_length)
for i=1:solver.trace_length
    q_targets[i] = r_batch[i] .+ (1.0 .- done_batch[i]).*discount(env.problem).*q_sp_max[i]
end

td_tracked = broadcast((x,y) -> x.*y, trace_mask_batch, q_sa - q_targets)
loss_tracked = loss.(td_tracked)
Flux.reset!(active_q)
Flux.truncate!(active_q)
Flux.reset!(target_q)
Flux.truncate!(target_q)


### DQN 

active_q = model 
target_q = deepcopy(model)



# s_batch is of size (intput_dims..., bs)
o = reset(env)
model(o[:])

model(flatten(s_batch))


q_values = active_q(flatten_batch(s_batch)) # n_actions x batch_size
q_sa = [q_values[a_batch[i], i] for i=1:batch_size]
q_sp_max = @view maximum(target_q(flatten_batch(sp_batch)), dims=1)[:]
q_targets = r_batch .+ (1.0 .- done_batch).*discount(env.problem).*q_sp_max # n_actions x batch_size

function loss(q_sa, q_targets)
    mean(huber_loss.(q_sa - q_targets))
end

l, td = loss(q_sa, q_targets)

Flux.data(l)

optimizer = ADAM(Flux.params(active_q), 1e-3)

# use deep copy to update the target network 

# use Flux.reset to reset RNN if necessary