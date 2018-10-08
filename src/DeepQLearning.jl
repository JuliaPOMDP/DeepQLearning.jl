module DeepQLearning

using Random
using StatsBase
using Flux
using Flux: onehot
using POMDPs
using POMDPModelTools
using DeepRL

export DeepQLearningSolver,
       AbstractNNPolicy,
       NNPolicy,
    
       # helpers
       flatten_batch,
       huber_loss


include("helpers.jl")
include("policy.jl")
include("exploration_policy.jl")
include("evaluation_policy.jl")
include("solver.jl")

end # module DeepQLearning