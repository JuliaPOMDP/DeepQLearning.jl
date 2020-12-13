module DeepQLearning

using Random
using StatsBase
using Printf
using Parameters
using Flux
using BSON
using POMDPModelTools
using POMDPPolicies
using POMDPLinter
using LinearAlgebra
using TensorBoardLogger: TBLogger, log_value
using EllipsisNotation

using CommonRLInterface: AbstractEnv, reset!, actions, observe, act!, terminated
import POMDPs
using POMDPs: MDP, POMDP, Policy, Solver, solve, action

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
