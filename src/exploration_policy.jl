#= 
interface for defining an exploration policy
=#

"""
    exploration(exp_policy, policy, env, obs, global_step, rng)
    return an action following an exploration policy 
    the use can provide its own exp_policy function
"""
function exploration(f::Function, policy::AbstractNNPolicy, env::AbstractEnvironment, obs, global_step::Int64, rng::AbstractRNG; reset_mask::Bool=false)
    return f(policy, env, obs, global_step, rng, reset_mask=reset_mask)
end

# Examples 

function linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64)
    function action_epsilon_greedy(policy::AbstractNNPolicy, env::AbstractEnvironment, obs, global_step::Int64, rng::AbstractRNG; reset_mask::Bool=false)
        eps = update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        if rand(rng) > eps 
            return (action(policy, obs), eps)
        else
            return (sample_action(env), eps)
        end
    end
    return action_epsilon_greedy
end

function update_epsilon(step::Int64, epsilon_fraction::Float64, epsilon_end::Float64, max_steps::Int64)
    new_eps = 0.    
    if step < epsilon_fraction*max_steps
        new_eps = 1 - (1 - epsilon_end)/(epsilon_fraction*max_steps)*step # decay
    else
        new_eps = epsilon_end
    end
    return new_eps
end


function fixed_epsilon_greedy(eps::Float64)
    function action_epsilon_greedy(policy::AbstractNNPolicy, env::AbstractEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        if rand(rng) > eps 
            return (action(policy, obs), eps)
        else
            return (sample_action(env), eps)
        end
    end
    return action_epsilon_greedy
end
