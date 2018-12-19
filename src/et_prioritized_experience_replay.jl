# Replay buffer that store full episodes

mutable struct ETPrioritizedEpisodeReplayBuffer
    max_size::Int64
    batch_size::Int64
    trace_length::Int64
    rng::AbstractRNG
    α::Float64
    β::Float64
    ϵ::Float64
    _curr_size::Int64
    _idx::Int64
    _experience::Vector{Vector{DQExperience}}
    _priorities::Vector{Array{Float64}}

    _s_batch::Vector{Array{Float64}}
    _a_batch::Vector{Array{Int64}}
    _r_batch::Vector{Array{Float64}}
    _sp_batch::Vector{Array{Float64}}
    _done_batch::Vector{Array{Bool}}
    _weights_batch::Vector{Array{Float64}}
    _trace_mask::Vector{Array{Int64}}
    _episode::Vector{DQExperience}
    _priorities_episode::Vector{Float64}

    function ETPrioritizedEpisodeReplayBuffer(env::AbstractEnvironment,
                          max_size::Int64,
                          batch_size::Int64,
                          trace_length::Int64,
                          rng::AbstractRNG = MersenneTwister(0),
                          α::Float64 = 0.6,
                          β::Float64 = 0.4,
                          ϵ::Float64 = 1e-3)
        s_dim = obs_dimensions(env)
        experience = Vector{Vector{DQExperience}}(undef, max_size)
        priorities = Vector{Array{Float64}}(undef, max_size)
        _s_batch = [zeros(s_dim..., batch_size) for i=1:trace_length]
        _a_batch = [zeros(Int64, batch_size) for i=1:trace_length]
        _r_batch = [zeros(batch_size) for i=1:trace_length]
        _sp_batch = [zeros(s_dim..., batch_size) for i=1:trace_length]
        _done_batch = [zeros(Bool, batch_size) for i=1:trace_length]
        _weights_batch = [zeros(Float64, batch_size) for i=1:trace_length]
        _trace_mask = [zeros(Int64, batch_size) for i=1:trace_length]
        _episode = Vector{DQExperience}()
        _priorities_episode = Vector{Float64}()
        return new(max_size, batch_size, trace_length, rng, α, β, ϵ, 0, 1, experience, priorities,
                   _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch, _weights_batch, _trace_mask, _episode, _priorities_episode)
    end
end

is_full(r::ETPrioritizedEpisodeReplayBuffer) = r._curr_size == r.max_size

max_size(r::ETPrioritizedEpisodeReplayBuffer) = r.max_size

function add_exp!(r::ETPrioritizedEpisodeReplayBuffer, exp::DQExperience, td_err::Float64=abs(exp.r))
    @assert td_err + r.ϵ > 0.0
    push!(r._episode, exp)
    push!(r._priorities_episode,  (td_err + r.ϵ)^r.α)
    if exp.done
        add_episode!(r)
        r._episode = Vector{DQExperience}()
        r._priorities_episode = Vector{Float64}()
    end
end

function add_episode!(r::ETPrioritizedEpisodeReplayBuffer)
    r._experience[r._idx] = r._episode
    r._priorities[r._idx] = r._priorities_episode
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

function reset_batches!(r::ETPrioritizedEpisodeReplayBuffer)
    fill!.(r._s_batch, 0.)
    fill!.(r._a_batch, 0)
    fill!.(r._r_batch, 0.)
    fill!.(r._sp_batch, 0.)
    fill!.(r._done_batch, false)
    fill!.(r._weights_batch, 0.)
    fill!.(r._trace_mask, 0)
end

function StatsBase.sample(r::ETPrioritizedEpisodeReplayBuffer)
    reset_batches!(r) # might not be necessary
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, Weights(sum.(r._priorities[1:r._curr_size])), r.batch_size, replace=false)
    @assert length(sample_indices) == size(r._s_batch[1])[end]
    start_indices = zeros(Int64, r.batch_size)
    for (i, idx) in enumerate(sample_indices)
        ep = r._experience[idx]
        pr = r._priorities[idx]
        # randomized start TODO add as an option of the buffer
        ep_start = rand(r.rng, 1:length(ep))
        start_indices[i] = ep_start
        t = 1
        for j=ep_start:min(length(ep), ep_start+r.trace_length-1)
            expe = ep[j]
            r._s_batch[t][axes(r._s_batch[t])[1:end-1]..., i] = expe.s
            r._a_batch[t][i] = expe.a
            r._r_batch[t][i] = expe.r
            r._sp_batch[t][axes(r._s_batch[t])[1:end-1]..., i] = expe.sp
            r._done_batch[t][i] = expe.done
            r._weights_batch[t][i] = pr[j]
            r._trace_mask[t][i] = 1
            t += 1
        end
    end
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, r._trace_mask, r._weights_batch, sample_indices, start_indices
end

function update_priorities!(r::ETPrioritizedEpisodeReplayBuffer, sample_indices::Vector{Int64}, start_indices::Vector{Int64}, td::Vector{Vector{Float64}})
    for i = 1:r.batch_size
        idx = sample_indices[i]
        ep_start = start_indices[i]
        ep = r._experience[idx]
        t = 1
        for j=ep_start:min(length(ep), ep_start+r.trace_length-1)
            r._priorities[idx][j] = (abs(td[t][i]) + r.ϵ)^r.α
            @assert r._priorities[idx][j] > 0.
            t = t+1
        end
    end
end

function populate_replay_buffer!(r::ETPrioritizedEpisodeReplayBuffer,
                                 env::AbstractEnvironment;
                                 max_pop::Int64 = r.max_size,
                                 max_steps::Int64 = 100)
    for t=1:(max_pop - r._curr_size)
        ep = generate_episode(env, max_steps=max_steps)
        add_episode!(r)
    end
    @assert r._curr_size >= r.batch_size
end

function generate_prioritized_episodeET(env::AbstractEnvironment; max_steps::Int64 = 100)
    episode = DQExperience[]
    sizehint!(episode, max_steps)
    # start simulation
    o = reset(env)
    done = false
    step = 1
    while !done && step < max_steps
        action = sample_action(env)
        ai = actionindex(env.problem, action)
        op, rew, done, info = step!(env, action)
        exp = DQExperience(o, ai, rew, op, done)
        push!(episode, exp)
        o = op
        step += 1
    end
    return episode
end
