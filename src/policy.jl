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

function POMDPs.action(policy::NNPolicy{P}, s::S) where {P <: MDP, S}
    action(policy, convert_s(Vector{Float64}, s, policy.problem))
end

function POMDPs.action(policy::NNPolicy{P}, o::O) where {P <: POMDP, O}
    action(policy, convert_o(Vector{Float64}, o, policy.problem))
end