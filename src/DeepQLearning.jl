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
using LinearAlgebra

export DeepQLearningSolver,
       AbstractNNPolicy,
       NNPolicy,
       getnetwork,
       resetstate!,
       DQExperience,
       restore_best_model,
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
include("prioritized_experience_replay.jl")
include("episode_replay.jl")
include("dueling.jl")
include("solver.jl")

end # module DeepQLearning
