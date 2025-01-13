# DeepQLearning

[![Build status](https://github.com/JuliaPOMDP/DeepQLearning.jl/workflows/CI/badge.svg)](https://github.com/JuliaPOMDP/DeepQLearning.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/JuliaPOMDP/DeepQLearning.jl/branch/master/graph/badge.svg?token=EfDZPMisVB)](https://codecov.io/github/JuliaPOMDP/DeepQLearning.jl)

This package provides an implementation of the Deep Q learning algorithm for solving MDPs. For more information see https://arxiv.org/pdf/1312.5602.pdf.
It uses POMDPs.jl and Flux.jl

It supports the following innovations:
- Target network
- Prioritized replay https://arxiv.org/pdf/1511.05952.pdf
- Dueling https://arxiv.org/pdf/1511.06581.pdf
- Double Q http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847
- Recurrent Q Learning

## Installation

```Julia
using Pkg
Pkg.add("DeepQLearning")
```

## Usage

```Julia
using DeepQLearning
using POMDPs
using Flux
using POMDPModels
using POMDPTools

# load MDP model from POMDPModels or define your own!
mdp = SimpleGridWorld();

# Define the Q network (see Flux.jl documentation)
# the gridworld state is represented by a 2 dimensional vector.
model = Chain(Dense(2, 32), Dense(32, length(actions(mdp))))

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2));

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

sim = RolloutSimulator(max_steps=30)
r_tot = simulate(sim, mdp, policy)
println("Total discounted reward for 1 simulation: $r_tot")
```

## Specifying exploration / evaluation policy

An exploration policy and evaluation policy can be specified in the solver parameters. 

An **exploration policy** can be provided in the form of a function that must return an action. The function provided will be called as follows: `f(policy, env, obs, global_step, rng)` where `policy` is the NN policy being trained, `env` the environment, `obs` the observation at which to take the action, `global_step` the interaction step of the solver, and `rng` a random number generator. This package provides by default an epsilon greedy policy with linear decrease of epsilon with `global_step`. 

An **evaluation policy** can be provided in a similar manner. The function will be called as follows: `f(policy, env, n_eval, max_episode_length, verbose)` where `policy` is the NN policy being trained, `env` the environment, `n_eval` the number of evaluation episode, `max_episode_length` the maximum number of steps in one episode, and `verbose` a boolean to enable printing or not. The evaluation function must returns three elements:
- Average total reward (Float), the average score per episode
- Average number of steps (Float), the average number of steps taken per episode
- Info, a dictionary mapping `String` to `Float` that can be used to log custom scalar values.

## Q-Network

The `qnetwork` options of the solver should accept any `Chain` object. It is expected that they will be multi-layer perceptrons or convolutional layers followed by dense layer. If the network is ending with dense layers, the `dueling` option will split all the dense layers at the end of the network. 

If the observation is a multi-dimensional array (e.g. an image), one can use the `flattenbatch` function to flatten all the dimensions of the image. It is useful to connect convolutional layers and dense layers for example. `flattenbatch` will flatten all the dimensions but the batch size. 

The input size of the network is problem dependent and must be specified when you create the q network.

This package exports the type `AbstractNNPolicy` which represents neural network based policy. In addition to the functions from `POMDPs.jl`, `AbstractNNPolicy` objects supports the following: 
    - `getnetwork(policy)`: returns the value network of the policy 
    - `resetstate!(policy)`: reset the hidden states of a policy (does nothing if it is not an RNN)

## Saving/Reloading model 

See [Flux.jl documentation](http://fluxml.ai/Flux.jl/stable/saving.html) for saving and loading models. The DeepQLearning solver saves the weights of the Q-network as a `bson` file in `solver.logdir/"qnetwork.bson"`.

## Logging

Logging is done through [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl). A log directory can be specified in the solver options, to disable logging you can set the `logdir` option to `nothing`.

## GPU Support

`DeepQLearning.jl` should support running the calculations on GPUs through the package [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl).
You must checkout the branch `gpu-support`. Note that it has not been tested thoroughly.
To run the solver on GPU you must first load `CuArrays` and then proceed as usual.

```julia
using CuArrays
using DeepQLearning
using POMDPs
using Flux
using POMDPModels

mdp = SimpleGridWorld();

# the model weights will be send to the gpu in the call to solve
model = Chain(Dense(2, 32), Dense(32, length(actions(mdp))))

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2));

solver = DeepQLearningSolver(qnetwork=model, max_steps=10000, 
                            exploration_policy=exploration,
                            learning_rate=0.005,log_freq=500,
                            recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)
```

## Solver Options

**Fields of the Q Learning solver:**
- `qnetwork::Any = nothing` Specify the architecture of the Q network 
- `exploration_policy::<ExplorationPolicy` Exploration strategy (e.g. EpsGreedyPolicy)
- `learning_rate::Float64 = 1e-4` learning rate 
- `max_steps::Int64` total number of training step default = 1000
- `batch_size::Int64` batch size sampled from the replay buffer default = 32
- `train_freq::Int64` frequency at which the active network is updated default  = 4
- `eval_freq::Int64` frequency at which to eval the network default = 100
- `target_update_freq::Int64` frequency at which the target network is updated default = 500
- `num_ep_eval::Int64` number of episodes to evaluate the policy default = 100
- `double_q::Bool` double q learning udpate default = true
- `dueling::Bool` dueling structure for the q network default = true
- `recurrence::Bool = false` set to true to use DRQN, it will throw an error if you set it to false and pass a recurrent model.
- `evaluation_policy::Function = basic_evaluation` function use to evaluate the policy every `eval_freq` steps, the default is a rollout that return the undiscounted average reward 
- `prioritized_replay::Bool` enable prioritized experience replay default = true
- `prioritized_replay_alpha::Float64` default = 0.6
- `prioritized_replay_epsilon::Float64` default = 1e-6
- `prioritized_replay_beta::Float64` default = 0.4
- `buffer_size::Int64` size of the experience replay buffer default = 1000
- `max_episode_length::Int64` maximum length of a training episode default = 100
- `train_start::Int64` number of steps used to fill in the replay buffer initially default = 200
- `rng::AbstractRNG` random number generator default = MersenneTwister(0)
- `logdir::String = ""` folder in which to save the model
- `save_freq::Int64` save the model every `save_freq` steps, default = 1000
- `log_freq::Int64` frequency at which to logg info default = 100
- `verbose::Bool` default = true
