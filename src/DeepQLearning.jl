module DeepQLearning

using Random
using StatsBase
using Printf
using Parameters
using Flux
using BSON
using POMDPs
using POMDPModelTools
using POMDPPolicies
using RLInterface

export DeepQLearningSolver,
       ETDeepQLearningSolver,
       DoubleDeepQLearningSolver,
       AbstractNNPolicy,
       NNPolicy,
       DQExperience,
       ETDQExperience,
       restore_best_model,
       ReplayBuffer,
       PrioritizedReplayBuffer,
       EpisodeReplayBuffer,
       ETReplayBuffer,
       ETEpisodeReplayBuffer,
       ETPrioritizedEpisodeReplayBuffer,

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
include("et_experience_replay.jl")
include("et_episode_replay.jl")
include("et_prioritized_experience_replay.jl")
include("dueling.jl")
include("solver.jl")
include("et_solver.jl")
include("double_solver.jl")

end # module DeepQLearning
