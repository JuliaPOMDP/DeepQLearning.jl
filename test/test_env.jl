using POMDPs
using POMDPModelTools

# Define a test environment
# it has 2 states, it ends up after taking 5 action
# Visiting the second state multiply the next reward by 10
# Optimal value 2.1
# Optimal Policy [2,1,2,1,3]

mutable struct TestMDP <: MDP{Tuple{Vector{Int32}, Int32}, Int64}
    shape
    stack::Int32
    o_stack::Int32
    max_time::Int32
    bad_state::Array{Int32}
    normal_state::Array{Int32}
    good_state::Array{Int32}
    _observation_space::Array{Array{Float32}}
    _rewards::Array{Float32}
    discount_factor::Float32
end

# RLInterface.obsvector_type(mdp::TestMDP) = Array{Float32, length(mdp.shape) + 1}

function TestMDP(shape=(6,), stack=4, max_time=6, discount_factor=0.99)
    bad_state =  convert.(Int32, rand(1:50, shape...))
    normal_state =  convert.(Int32, rand(100:150, shape...))
    good_state =  convert.(Int32, rand(150:200, shape...))
    _observation_space = [bad_state, normal_state, good_state]
    _rewards = [-0.1f0, 0.0f0, 0.1f0]
    return TestMDP(shape, 4, stack, max_time, bad_state, normal_state, good_state, _observation_space, _rewards, discount_factor)
end

POMDPs.discount(mdp::TestMDP) = mdp.discount_factor

function POMDPs.observations(mdp::TestMDP)
    return mdp._observation_space
end

function POMDPs.actions(mdp::TestMDP)
    return 1:4
end

POMDPs.actionindex(mdp::TestMDP, a::T) where T<:Integer = a

function POMDPs.initialstate(mdp::TestMDP)
    ImplicitDistribution() do rng
        init_t = Int32(1)
        init_s = fill(Int32(1), mdp.stack)
        return (init_s, init_t)
    end
end

function POMDPs.convert_s(t::Type{V}, s::Tuple{Vector{T}, T}, mdp::TestMDP) where {T,V<:AbstractArray}
    obs = zeros(mdp.shape..., mdp.o_stack)
    for i=1:mdp.o_stack
        obs[Base.setindex(axes(obs), i, ndims(obs))...] = observations(mdp)[s[1][end-i+1]]
    end
    return convert(t, obs./255.0f0)
end

function was_in_second(s::Tuple{Vector{T}, T}) where T<:Integer
    s[1][end] == 2
end

function POMDPs.gen(mdp::TestMDP, s::Tuple{Vector{T}, T}, a::N, rng::AbstractRNG) where {T <: Integer, N <: Integer}
    t_new = s[2] + convert(Int32, 1) # increment time
    s_new = circshift(s[1], -1)
    if a < 4
        s_new[end] = a
    else
        s_new[end] = s_new[end-1]
    end
    return (sp=(s_new, t_new),)
end

function POMDPs.reward(mdp::TestMDP, s::Tuple{Vector{T}, T}, a::N, sp::Tuple{Vector{T}, T}) where {T <: Integer, N <: Integer}
    r = mdp._rewards[sp[1][end]]
    if was_in_second(s)
        r *= -10
    end
    return r
end

function POMDPs.isterminal(mdp::TestMDP, s)
    return s[2] >= mdp.max_time
end
