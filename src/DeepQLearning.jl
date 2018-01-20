module DeepQLearning

# package code goes here
using Distributions, StatsBase, Parameters
using TensorFlow
using POMDPs, POMDPToolbox, DeepRL
using JLD
const tf = TensorFlow

export DeepQLearningSolver,
       QNetworkArchitecture,
       # tf helpers
       flatten,
       dense,
       conv2d,
       mlp,
       cnn_to_mlp,
       get_train_vars_by_name,

       # replay buffer
       DQExperience,
       ReplayBuffer,
       is_full,
       max_size,
       add_exp!,
       populate_replay_buffer!,

       # training
       TrainGraph,
       DQNPolicy,
       action,
       solve,
       save,
       restore

include("tf_helpers.jl")
include("experience_replay.jl")

"""
Specify the Q network architecture
[(int, (int,int), int)]
"""
@with_kw mutable struct QNetworkArchitecture
   fc::Vector{Int64} = Vector{Int64}[]
   conv::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
end

"""
    Deep Q Learning Solver type
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
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    verbose::Bool = true
end

include("graph.jl")
include("policy.jl")
include("q_network.jl")
include("solver.jl")
include("saver.jl")
end # module
