"""
    flattenbatch(x::AbstractArray)
flatten a multi dimensional array to keep only the last dimension.
It returns a 2 dimensional array of size (flatten_dim, batch_size)
"""
function flattenbatch(x::AbstractArray)
    reshape(x, (:, size(x)[end]))
end

"""
    huber_loss(x)
Compute the Huber Loss (from ReinforcementLearning.jl)
"""
function huber_loss(x)
    abserror = abs.(x)
    quadratic = min.(abserror, one(x))
    linear = abserror .- quadratic 
    return 0.5f0 .*quadratic .* quadratic .+ linear
end

"""
    isrecurrent(m)
returns true if m contains a recurrent layer 
"""
function isrecurrent(m)
    for layer in m 
        if layer isa Flux.Recur 
            return true 
        end
    end
    return false
end

"""
    globalnorm(p::Params, gs::Flux.Zygote.Grads)
returns the maximum absolute values in the gradients of W 
"""
function globalnorm(ps::Flux.Params, gs::Flux.Zygote.Grads)
    gnorm = 0f0
    for p in ps 
        gs[p] === nothing && continue 
        curr_norm = maximum(abs.(gs[p]))
        gnorm =  curr_norm > gnorm  ? curr_norm : gnorm
    end 
    return gnorm
end

"""
    batch_trajectories(s::AbstractArray, traj_length::Int64, batch_size::Int64)
converts multidimensional arrays into batches of trajectories to be process by a Flux recurrent model. 
It takes as input an array of dimension state_dim... x traj_length x batch_size
"""
function batch_trajectories(s::AbstractArray, traj_length::Int64, batch_size::Int64) 
    return Flux.batchseq([[s[axes(s)[1:end-2]..., j, i] for j=1:traj_length] for i=1:batch_size], 0)
end

""" 
    hiddenstates(m)
returns the hidden states of all the recurrent layers of a model 
""" 
function hiddenstates(m)
    return [l.state for l in m if l isa Flux.Recur]
end

"""
    sethiddenstates!(m, hs)
Given a list of hiddenstate, set the hidden state of each recurrent layer of the model m 
to what is in the list. 
The order of the list should match the order of the recurrent layers in the model.
"""
function sethiddenstates!(m, hs)
    i = 1
    for l in m
        if isa(l, Flux.Recur) 
            l.state = hs[i]
            i += 1
        end
    end
end

obs_dimensions(env::AbstractEnv) = size(observe(env))

default_discount(env) = 1.0
default_discount(env::MDPCommonRLEnv) = POMDPs.discount(convert(MDP, env.m))
default_discount(env::POMDPCommonRLEnv) = POMDPs.discount(convert(POMDP, env.m))
