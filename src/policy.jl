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

function get_value(sess, q, o)
    o = reshape(o, (1, obs_dim...))
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

function get_value_batch(sess, q, o)
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

"""
Evaluate a Q network
"""
function eval_q(graph::TrainGraph,
                env::Union{MDPEnvironment, POMDPEnvironment};
                n_eval::Int64=100,
                max_episode_length::Int64=100)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= max_episode_length
            action =  get_action(graph, env, obs)
            # println(action)
            obs, rew, done, info = step!(env, action)
            # println("state ", env.state, " action ", a)
            # println("Reward ", rew)
            # println(obs, " ", done, " ", info, " ", step)
            r_tot += rew
            step += 1
        end
        avg_r += r_tot
        # println(r_tot)

    end
    return  avg_r /= n_eval
end
