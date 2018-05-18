__precompile__()

module DeepQLearning

# package code goes here
using Distributions, StatsBase, Parameters
using TensorFlow
using POMDPs, POMDPToolbox, DeepRL
using JLD
const tf = TensorFlow

export DeepQLearningSolver,
       QNetworkArchitecture,
       DeepRecurrentQLearningSolver,
       RecurrentQNetworkArchitecture,
       # tf helpers
       flatten,
       dense,
       conv2d,
       mlp,
       cnn_to_mlp,
       get_train_vars_by_name,
       init_session,
       logg_scalar,

       # replay buffer
       DQExperience,
       ReplayBuffer,
       PrioritizedReplayBuffer,
       is_full,
       max_size,
       add_exp!,
       populate_replay_buffer!,
       update_priorities!,

       # training
       TrainGraph,
       DQNPolicy,
       LSTMPolicy,
       reset_hidden_state!,
       get_action,
       get_action!,
       eval_q,
       eval_lstm,
       action,
       solve,
       save,
       restore

include("tf_helpers.jl")

abstract type AbstractNNPolicy <: Policy end

"""
    QNetworkArchitecture
Specify the Q network architecture as convolutional layers followed by a multi layer perceptron
- `fc::Vector{Int64}` the number of nodes and hidden layers [8, 32] => 2 layers of 8 and 32 nodes
- `conv::Vector{Tuple{Int64, Vector{Int64}, Int64}}` [(#filters, (kernel_size,kernel_size), stride)]
"""
@with_kw mutable struct QNetworkArchitecture
   fc::Vector{Int64} = Vector{Int64}[]
   conv::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
end

"""
    DeepQLearningSolver

## Fields:
- `arch::QNetworkArchitecture` Specify the architecture of the Q network default = QNetworkArchitecture(conv=[], fc=[])
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
"""
@with_kw mutable struct DeepQLearningSolver
    arch::QNetworkArchitecture = QNetworkArchitecture(conv=[], fc=[])
    lr::Float64 = 0.005
    max_steps::Int64 = 1000
    target_update_freq::Int64 = 500
    batch_size::Int64 = 32
    train_freq::Int64  = 4
    log_freq::Int64 = 100
    eval_freq::Int64 = 100
    num_ep_eval::Int64 = 100
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    double_q::Bool = true
    dueling::Bool = true
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    logdir::String = "log"
    save_freq::Int64 = 10000
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end)
    verbose::Bool = true
end

include("recurrent_deep_q_learning.jl")
include("experience_replay.jl")
include("episode_replay.jl")
include("prioritized_experience_replay.jl")
include("policy.jl")
include("exploration_policy.jl")
include("graph.jl")
include("q_network.jl")
include("solver.jl")
include("saver.jl")
end # module
