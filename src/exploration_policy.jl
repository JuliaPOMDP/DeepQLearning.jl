#= 
interface for defining an exploration policy
=#

"""
    exploration(exp_policy, policy, env, obs, global_step, rng)
    return an action following an exploration policy 
    the use can provide its own exp_policy function
"""
function exploration(f::Function, policy::AbstractNNPolicy, env::AbstractEnv, obs, global_step::Int64, rng::AbstractRNG)
    return f(policy, env, obs, global_step, rng)
end

# Examples 

function linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float32, eps_end::Float32)
    function action_epsilon_greedy(policy::AbstractNNPolicy, env::AbstractEnv, obs, global_step::Int64, rng::AbstractRNG)
        eps = update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        if rand(rng) > eps 
            return (action(policy, obs), eps)
        else
            return (rand(actions(env)), eps)
        end
    end
    return action_epsilon_greedy
end

function update_epsilon(step::Int64, epsilon_fraction::Float32, epsilon_end::Float32, max_steps::Int64)
    new_eps = 0.    
    if step < epsilon_fraction*max_steps
        new_eps = 1 - (1 - epsilon_end)/(epsilon_fraction*max_steps)*step # decay
    else
        new_eps = epsilon_end
    end
    return new_eps
end
