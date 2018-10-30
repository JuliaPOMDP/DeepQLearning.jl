module DeepQLearning

using Random
using StatsBase
using Printf
using Parameters
using Flux
using BSON
using POMDPs
using POMDPModelTools
using DeepRL

export DeepQLearningSolver,
       AbstractNNPolicy,
       NNPolicy,
       DQExperience,
       ReplayBuffer,
       PrioritizedReplayBuffer,
       EpisodeReplayBuffer,

    
       # helpers
       flattenbatch,
       huber_loss,
       isrecurrent,
       batch_trajectories


include("helpers.jl")
include("policy.jl")
include("exploration_policy.jl")
include("evaluation_policy.jl")
include("experience_replay.jl")
include("prioritized_experience_replay.jl")
include("episode_replay.jl")
include("dueling.jl")
include("solver.jl")

end # module DeepQLearning