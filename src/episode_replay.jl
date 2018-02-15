# Replay buffer that store full episodes

mutable struct EpisodeReplayBuffer
    max_size::Int64
    batch_size::Int64
    trace_length::Int64
    rng::AbstractRNG
    _curr_size::Int64
    _idx::Int64
    _experience::Vector{Vector{DQExperience}}

    _s_batch::Array{Float64}
    _a_batch::Array{Int64}
    _r_batch::Array{Float64}
    _sp_batch::Array{Float64}
    _done_batch::Array{Bool}
    _trace_mask::Array{Int64}

    function EpisodeReplayBuffer(env::AbstractEnvironment,
                          max_size::Int64,
                          batch_size::Int64,
                          trace_length::Int64,
                          rng::AbstractRNG = MersenneTwister(0))
        s_dim = obs_dimensions(env)
        experience = Vector{Vector{DQExperience}}(max_size)
        _s_batch = zeros(batch_size, trace_length, s_dim...)
        _a_batch = zeros(Int64, batch_size, trace_length)
        _r_batch = zeros(batch_size, trace_length)
        _sp_batch = zeros(batch_size, trace_length, s_dim...)
        _done_batch = zeros(Bool, batch_size, trace_length)
        _trace_mask = zeros(Int64, batch_size, trace_length)
        return new(max_size, batch_size, trace_length, rng, 0, 1, experience,
                   _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch, _trace_mask)
    end
end

is_full(r::EpisodeReplayBuffer) = r._curr_size == r.max_size

max_size(r::EpisodeReplayBuffer) = r.max_size

function add_episode!(r::EpisodeReplayBuffer, ep::Vector{DQExperience})
    r._experience[r._idx] = ep
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

function reset_batches!(r::EpisodeReplayBuffer)
    fill!(r._s_batch, 0.)
    fill!(r._a_batch, 0)
    fill!(r._r_batch, 0.)
    fill!(r._sp_batch, 0.)
    fill!(r._done_batch, false)
    fill!(r._trace_mask, 0)
end

function StatsBase.sample(r::EpisodeReplayBuffer)
    reset_batches!(r) # might not be necessary
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, r.batch_size, replace=false)
    @assert length(sample_indices) == size(r._s_batch)[1]
    for (i, idx) in enumerate(sample_indices)
        ep = r._experience[idx]
        # randomized start TODO add as an option of the buffer
        ep_start = rand(r.rng, 1:length(ep))
        t = 1
        for j=ep_start:min(length(ep), r.trace_length)
            expe = ep[j]
            r._s_batch[i,t,indices(r._s_batch)[3:end]...] = expe.s
            r._a_batch[i,t] = expe.a
            r._r_batch[i,t] = expe.r
            r._sp_batch[i,t,indices(r._sp_batch)[3:end]...] = expe.sp
            r._done_batch[i,t] = expe.done
            r._trace_mask[i,t] = 1
            t += 1
        end
    end
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, r._trace_mask
end


function populate_replay_buffer!(r::EpisodeReplayBuffer,
                                 env::AbstractEnvironment;
                                 max_pop::Int64 = r.max_size,
                                 max_steps::Int64 = 100)
    for t=1:(max_pop - r._curr_size)
        ep = generate_episode(env, max_steps=max_steps)
        add_episode!(r, ep)
    end
    @assert r._curr_size >= r.batch_size
end

function generate_episode(env::AbstractEnvironment; max_steps::Int64 = 100)
    episode = DQExperience[]
    sizehint!(episode, max_steps)
    # start simulation
    o = reset(env)
    done = false
    step = 1
    while !done && step < max_steps
        action = sample_action(env)
        ai = action_index(env.problem, action)
        op, rew, done, info = step!(env, action)
        exp = DQExperience(o, ai, rew, op, done)
        push!(episode, exp)
        o = op
        step += 1
    end
    return episode
end
