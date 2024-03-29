# Naive implementation

struct DQExperience{N <: Real,T <: Real, A<:AbstractArray}
    s::A
    a::N
    r::T
    sp::A
    done::Bool
end

function Base.convert(::Type{DQExperience{Int32, Float32, C}}, x::DQExperience{A, B, C}) where {A, B, C}
    return DQExperience{Int32, Float32, C}(convert(C, x.s),
                                            convert(Int32, x.a),
                                            convert(Float32, x.r),
                                            convert(C, x.sp),
                                            x.done)
end

mutable struct PrioritizedReplayBuffer{N<:Integer, T<:AbstractFloat,CI,Q,A<:AbstractArray{T}}
    max_size::Int64
    batch_size::Int64
    rng::AbstractRNG
    α::Float32
    β::Float32
    ϵ::Float32
    _curr_size::Int64
    _idx::Int64
    _priorities::Vector{T}
    _experience::Vector{DQExperience{N,T,Q}}

    _s_batch::A
    _a_batch::Vector{CI}
    _r_batch::Vector{T}
    _sp_batch::A
    _done_batch::Vector{T}
    _weights_batch::Vector{T}
end

function PrioritizedReplayBuffer(env::AbstractEnv,
                                max_size::Int64,
                                batch_size::Int64;
                                rng::AbstractRNG = MersenneTwister(0),
                                α::Float32 = 6f-1,
                                β::Float32 = 4f-1,
                                ϵ::Float32 = 1f-3)
    o = observe(env)
    s_dim = size(o)
    experience = Vector{DQExperience{Int32, Float32, typeof(o)}}(undef, max_size)
    priorities = Vector{Float32}(undef, max_size)
    _s_batch = zeros(Float32, s_dim..., batch_size)
    _a_batch = [CartesianIndex(0,0) for i=1:batch_size]
    _r_batch = zeros(Float32, batch_size)
    _sp_batch = zeros(Float32, s_dim..., batch_size)
    _done_batch = zeros(Float32, batch_size)
    _weights_batch = zeros(Float32, batch_size)
    return PrioritizedReplayBuffer(max_size, batch_size, rng, α, β, ϵ, 0, 1, priorities, experience,
                _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch, _weights_batch)
end


is_full(r::PrioritizedReplayBuffer) = r._curr_size == r.max_size

max_size(r::PrioritizedReplayBuffer) = r.max_size

function add_exp!(r::PrioritizedReplayBuffer, expe::DQExperience, td_err::T=abs(expe.r)) where T
    @assert td_err + r.ϵ > 0.
    priority = (td_err + r.ϵ)^r.α
    r._experience[r._idx] = expe
    r._priorities[r._idx] = priority
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

function update_priorities!(r::PrioritizedReplayBuffer, indices::Vector{Int64}, td_errors::V) where V <: AbstractArray
    new_priorities = (abs.(td_errors) .+ r.ϵ).^r.α
    @assert all(new_priorities .> 0f0)
    r._priorities[indices] = new_priorities
end

function StatsBase.sample(r::PrioritizedReplayBuffer)
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, Weights(r._priorities[1:r._curr_size]), r.batch_size, replace=false)
    return get_batch(r, sample_indices)
end

function get_batch(r::PrioritizedReplayBuffer, sample_indices::Vector{Int64})
    @assert length(sample_indices) == size(r._s_batch)[end]
    for (i, idx) in enumerate(sample_indices)
        @inbounds begin
            r._s_batch[.., i] = vec(r._experience[idx].s)
            r._a_batch[i] = CartesianIndex(r._experience[idx].a, i)
            r._r_batch[i] = r._experience[idx].r
            r._sp_batch[.., i] = vec(r._experience[idx].sp)
            r._done_batch[i] = r._experience[idx].done
            r._weights_batch[i] = r._priorities[idx]
        end
    end
    p = r._weights_batch ./ sum(r._priorities[1:r._curr_size])
    weights = (r._curr_size * p).^(-r.β)
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, sample_indices, weights
end

function populate_replay_buffer!(replay::PrioritizedReplayBuffer,
                                 env::AbstractEnv,
                                 action_indices;
                                 max_pop::Int64=replay.max_size, max_steps::Int64=100,
                                 policy::Policy = FunctionPolicy(o->rand(actions(env))))
    reset!(env)
    o = observe(env)
    done = false
    step = 0
    for t=1:(max_pop - replay._curr_size)
        a = action(policy, o)
        ai = action_indices[a]
        rew = act!(env, a)
        op = observe(env)
        done = terminated(env)
        exp = DQExperience(o, ai, Float32(rew), op, done)
        add_exp!(replay, exp, abs(Float32(rew))) # assume initial td error is r
        o = op
        # println(o, " ", action, " ", rew, " ", done, " ", info) #TODO verbose?
        step += 1
        if done || step >= max_steps
            reset!(env)
            o = observe(env)
            done = false
            step = 0
        end
    end
    @assert replay._curr_size >= replay.batch_size
end
