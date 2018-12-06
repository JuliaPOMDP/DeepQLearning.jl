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
    avg_violations = 0
    avg_steps = 0
    for i=1:n_eval
        done = false 
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset!(policy)
        while !done && step <= max_episode_length
            act = action(policy, obs)
            obs, rew, done, info = step!(env, act)
            r_tot += rew 
            step += 1
        end
        r_tot < 0.1 ? avg_violations += 1 : nothing
        avg_r += r_tot
        avg_steps += step 
    end
    avg_r /= n_eval
    avg_violations = avg_violations / n_eval * 100
    avg_steps /= n_eval
    if verbose
        println("Evaluation ... Avg Reward $(avg_r) | Avg Violations (%) $(avg_violations) | Avg Steps $(avg_steps). ")
    end
    return  avg_r, avg_violations, avg_steps
end
