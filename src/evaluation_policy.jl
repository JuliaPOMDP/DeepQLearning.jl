#= 
Interface for defining an evaluation policy 
=#

"""
    evaluation(eval_policy, policy, env, obs, global_step, rng)
    returns the average reward of the current policy, the user can specify its own function 
    f to carry the evaluation, we provide a default basic_evaluation that is just a rollout. 
"""
function evaluation(f::Function, policy::AbstractNNPolicy, env::AbstractEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool = false)
    return f(policy, env, n_eval, max_episode_length, verbose)
end


# Examples  
# just simulate the policy, return the average non discounted reward
function basic_evaluation(policy::AbstractNNPolicy, env::AbstractEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
    avg_r = 0 
    for i=1:n_eval
        done = false 
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset_hidden_state!(policy)
        while !done && step <= max_episode_length
            act = get_action!(policy, obs)
            obs, rew, done, info = step!(env, act)
            r_tot += rew 
            step += 1
        end
        avg_r += r_tot 
    end
    if verbose
        println("Evaluation ... Avg Reward ", avg_r/n_eval)
    end
    return  avg_r /= n_eval
end
