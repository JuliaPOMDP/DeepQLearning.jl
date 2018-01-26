# DeepQLearning

[![Build Status](https://travis-ci.org/MaximeBouton/DeepQLearning.jl.svg?branch=master)](https://travis-ci.org/MaximeBouton/DeepQLearning.jl)

[![Coverage Status](https://coveralls.io/repos/MaximeBouton/DeepQLearning.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/MaximeBouton/DeepQLearning.jl?branch=master)

[![codecov.io](http://codecov.io/github/MaximeBouton/DeepQLearning.jl/coverage.svg?branch=master)](http://codecov.io/github/MaximeBouton/DeepQLearning.jl?branch=master)

Implementation of the DeepQLearning algorithm in Julia using TensorFlow.jl and POMDPs.jl.
This package provide an implementation of the Deep Q learning algorithm. For more information see https://arxiv.org/pdf/1312.5602.pdf

It supports the following innovations:
- Target network
- Prioritized replay https://arxiv.org/pdf/1511.05952.pdf
- Dueling https://arxiv.org/pdf/1511.06581.pdf
- Double Q http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847

## Installation

```Julia
Pkg.clone()
```

## Usage

```Julia
using DeepQLearning
using POMDPModels
using POMDPToolbox

# define a solver with a 32x8 fully connected NN to describe the Q values
# uses double q learning and dueling
solver = DeepQLearning(max_steps = 100000,
                       lr = 0.005,
                       target_update_freq = 1000,
                       arch = QNetworkArchitecture(conv=[], fc=[32,8]),
                       double_q = true,
                       dueling = true)
mdp = GridWorld()
policy = solve(solver, mdp)

sim = RolloutSimulator(max_steps=30)
r_tot = simulate(sim, mdp, policy)
println("Total discounted reward for 1 simulation: $r_tot")
```

**Fields:**
- `arch::QNetworkArchitecture`Specify the architecture of the Q network default = QNetworkArchitecture(conv=[], fc=[])
- `lr::Float64` learning rate default = 0.005
- `max_steps::Int64` total number of training step default = 1000
- `target_update_freq::Int64` frequency at which the target network is updated default = 500
- `batch_size::Int64` batch size sampled from the replay buffer default = 32
- `train_freq::Int64` frequency at which the active network is updated default  = 4
- `log_freq::Int64` frequency at which to logg info default = 100
- `eval_freq::Int64` frequency at which to eval the network default = 100
- `num_ep_eval::Int64` number of episodes to evaluate the policy default = 100
- `eps_fraction::Float64` fraction of the training set used to explore default = 0.5
- `eps_end::Float64` value of epsilon at the end of the exploration phase default = 0.01
- `double_q::Bool` double q learning udpate default = true
- `dueling::Bool` dueling structure for the q network default = true
- `prioritized_replay::Bool` enable prioritized experience replay default = true
- `prioritized_replay_alpha::Float64` default = 0.6
- `prioritized_replay_epsilon::Float64` default = 1e-6
- `prioritized_replay_beta::Float64` default = 0.4
- `buffer_size::Int64` size of the experience replay buffer default = 1000
- `max_episode_length::Int64` maximum length of a training episode default = 100
- `train_start::Int64` number of steps used to fill in the replay buffer initially default = 200
- `grad_clip::Bool` enables gradient clipping default = true
- `clip_val::Float64` maximum value for the grad norm default = 10.0
- `rng::AbstractRNG` random number generator default = MersenneTwister(0)
- `verbose::Bool` default = true


## Work in progress

- Deep Recurrent Q Network
- Policy correction
