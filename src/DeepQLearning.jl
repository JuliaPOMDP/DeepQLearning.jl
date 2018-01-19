module DeepQLearning

# package code goes here
using Distributions, StatsBase, Parameters
using TensorFlow
using POMDPs, POMDPToolbox, DeepRL
const tf = TensorFlow

export DeepQLearningSolver,
       QNetworkArchitecture
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
       




include("tf_helpers.jl")
include("experience_replay.jl")
include("policy.jl")
include("q_network.jl")


end # module
