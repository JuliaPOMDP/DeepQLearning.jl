abstract type AbstractNNPolicy  <: Policy end

struct NNPolicy{P <: Union{MDP, POMDP}, Q, A} <: AbstractNNPolicy 
    problem::P
    qnetwork::Q
    action_map::Vector{A}
    n_input_dims::Int64
end

function reset!(policy::NNPolicy)
    Flux.reset!(policy.qnetwork)
end

function action(policy::NNPolicy, o::AbstractArray{T}) where T<:Real
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        vals = policy.qnetwork(obatch)
        return policy.action_map[argmax(vals)]
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end