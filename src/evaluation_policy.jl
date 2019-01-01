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
    avg_timeout = 0
    for i=1:n_eval
        done = false 
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset!(policy)
        while !done && step <= max_episode_length
            act = action(policy, obs)
            obs, rew, done, info = step!(env, act)
            r_tot = discount(env.problem)^step*rew + r_tot 
            step += 1
        end
        if r_tot < 0.5
            if step > max_episode_length
                avg_timeout += 1
            else
                avg_violations += 1
            end 
        end
        avg_r += r_tot
        avg_steps += step 
    end
    avg_r /= n_eval
    avg_violations = avg_violations / n_eval * 100
    avg_steps /= n_eval
    avg_timeout = avg_timeout / n_eval * 100
    if verbose
        println("Evaluation ... Avg Reward $(avg_r) | Avg Violations (%) $(avg_violations) | Avg Steps $(avg_steps) | Avg Timeout (%) $(avg_timeout). ")
    end
    return  avg_r, avg_violations, avg_steps, avg_timeout
end
