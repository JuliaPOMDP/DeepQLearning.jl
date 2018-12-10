
struct ETDQExperience
    s::Array{Float64}
    a::Int64
    q::Float64
end

mutable struct ETReplayBuffer
    max_size::Int64 # maximum size of the buffer
    batch_size::Int64
    rng::AbstractRNG
    _curr_size::Int64
    _idx::Int64
    _experience::Vector{ETDQExperience}
    _gamma::Float64
    _lambda::Float64

    _s_dim::Tuple{Int64}
    _s_batch::Array{Float64}
    _a_batch::Vector{Int64}
    _q_batch::Vector{Float64}
    _episode::Vector{DQExperience}

    function ETReplayBuffer(env::AbstractEnvironment,
                          max_size::Int64,
                          batch_size::Int64,
                          gamma::Float64,
                          lambda::Float64,
                          rng::AbstractRNG = MersenneTwister(0))
        _s_dim = obs_dimensions(env)
        experience = Vector{ETDQExperience}(undef, max_size)
        _s_batch = zeros(_s_dim..., batch_size)
        _a_batch = zeros(Int64, batch_size)
        _q_batch = zeros(batch_size)
        _episode = Vector{DQExperience}()
        return new(max_size, batch_size, rng, 0, 1, experience, gamma, lambda, _s_dim,
                   _s_batch, _a_batch, _q_batch, _episode)
    end
end

is_full(r::ETReplayBuffer) = r._curr_size == r.max_size

max_size(r::ETReplayBuffer) = r.max_size

function add_exp!(r::ETReplayBuffer, expe::DQExperience, active_q::Any, target_q::Any, double_q::Bool)
    push!(r._episode, expe)
    if expe.done
        add_episode!(r, active_q, target_q, double_q)
        r._episode = Vector{DQExperience}()
    end    
    """
    r._experience[r._idx] = expe
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
    """
end


function add_episode!(r::ETReplayBuffer, active_q::Any, target_q::Any, double_q::Bool)
    N = length(r._episode)
    s_batch = zeros(r._s_dim..., N)
    a_batch = zeros(Int64, N)
    r_batch = zeros(Int64, N)
    sp_batch = zeros(r._s_dim..., N)
    for i = 1:N
        s_batch[Base.setindex(axes(s_batch), i, ndims(s_batch))...]  = r._episode[i].s
        a_batch[i] = r._episode[i].a
        r_batch[i] = r._episode[i].r
        sp_batch[Base.setindex(axes(sp_batch), i, ndims(sp_batch))...] = r._episode[i].sp
    end

    q_targets = get_target_q(active_q, target_q, double_q, s_batch, a_batch, r_batch, sp_batch,r._gamma, r._lambda)
    for i = 1:N
        r._experience[r._idx] = ETDQExperience(r._episode[i].s, a_batch[i], q_targets[i])
        r._idx = mod1((r._idx+1), r.max_size)
        if r._curr_size < r.max_size
            r._curr_size += 1
        end
    end

end


function StatsBase.sample(r::ETReplayBuffer)
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, r.batch_size, replace=false)
    return get_batch(r, sample_indices)
end

function get_batch(r::ETReplayBuffer, sample_indices::Vector{Int64})
    @assert length(sample_indices) == size(r._s_batch)[end]
    for (i, idx) in enumerate(sample_indices)
        r._s_batch[Base.setindex(axes(r._s_batch), i, ndims(r._s_batch))...] = r._experience[idx].s
        r._a_batch[i] = r._experience[idx].a
        r._q_batch[i] = r._experience[idx].q
    end
    return r._s_batch, r._a_batch, r._q_batch
end


function populate_replay_buffer!(replay::ETReplayBuffer, env::AbstractEnvironment, active_q, target_q, double_q;
                                 max_pop::Int64=replay.max_size, max_steps::Int64=100)
    o = reset(env)
    done = false
    step = 0
    for t=1:(max_pop - replay._curr_size)
        action = sample_action(env)
        ai = actionindex(env.problem, action)
        op, rew, done, info = step!(env, action)
        exp = DQExperience(o, ai, rew, op, done)
        add_exp!(replay, exp, active_q, target_q, double_q)
        o = op
        # println(o, " ", action, " ", rew, " ", done, " ", info) #TODO verbose?
        step += 1
        if done || step >= max_steps
            o = reset(env)
            done = false
            step = 0
        end
    end
    @assert replay._curr_size >= replay.batch_size
end
