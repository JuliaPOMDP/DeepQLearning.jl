mutable struct DQNPolicy <: Policy
    q::Tensor # Q network
    s::Tensor # placeholder
    env::AbstractEnvironment
    sess
end


function get_action(graph::TrainGraph, env::Union{MDPEnvironment, POMDPEnvironment}, o::Array{Float64})
    # cannot take a batch of observations
    o = reshape(o, (1, size(o)...))
    q_val = run(graph.sess, graph.q, Dict(graph.s => o) )
    ai = indmax(q_val)
    return actions(env)[ai] # inefficient
end

function get_action(policy::DQNPolicy, o::Array{Float64})
    # cannot take a batch of observations
    o = reshape(o, (1, size(o)...))
    q_val = run(policy.sess, policy.q, Dict(policy.s => o))
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end


function get_value(sess, q, o)
    o = reshape(o, (1, obs_dim...))
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

function get_value_batch(sess, q, o)
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

function POMDPs.action(policy::DQNPolicy, s::S) where S
    obs = nothing
    if isa(policy.env.problem, POMDP)
        obs = convert_o(Vector{Float64}, s, policy.env.problem)
    else
        obs = convert_s(Vector{Float64}, s, policy.env.problem)
    end
    return get_action(policy, obs)
end
