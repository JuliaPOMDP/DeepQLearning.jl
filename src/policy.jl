mutable struct DQNPolicy <: Policy
    q::Tensor # Q network
    env::AbstractEnvironment
    sess
end


function get_action(sess, env, q, o)
    # cannot take a batch of observations
    o = reshape(o, (1, obs_dim...))
    q_val = run(sess, q, Dict(s => o) )
    ai = indmax(q_val)
    return actions(env)[ai] # inefficient
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
