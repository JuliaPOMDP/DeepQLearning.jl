using Random
using StatsBase
using Flux
using Flux: onehot
using POMDPs
using DeepRL

include("../src/experience_replay.jl")

include("test_env.jl")

mdp = TestMDP((5,5), 4, 6)


### Hyperparameters
buffer_size = 3000
batch_size = 32
train_start = 200
learning_rate = 1e-4

env = MDPEnvironment(mdp)

### MODEL BUILDING 
input_dims = obs_dimensions(env)
flattened_input_dims = reduce(*, input_dims)
output_dims = n_actions(env)
model = Chain(Dense(flattened_input_dims, 32), Dense(32, output_dims))

### Replay Buffer (unchanged from previous implementation)
replay = ReplayBuffer(env, buffer_size, batch_size)
populate_replay_buffer!(replay, env, max_pop=train_start)


### Batch Train, in Flux, the batch size is the last dimension
s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)


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

optimizer = ADAM(Flux.params(active_q), 1e-3)

# use deep copy to update the target network 

# use Flux.reset to reset RNN if necessary