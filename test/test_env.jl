using POMDPs
using POMDPModelTools
using DeepRL

# Define a test environment
# it has 2 states, it ends up after taking 5 action
# Visiting the second state multiply the next reward by 10
# Optimal value 2.1
# Optimal Policy [2,1,2,1,3]

mutable struct TestMDP <: MDP{Tuple{Vector{Int64}, Int64}, Int64}
    shape
    stack::Int64
    o_stack::Int64
    max_time::Int64
    bad_state::Array{Int64}
    normal_state::Array{Int64}
    good_state::Array{Int64}
    _observation_space::Array{Array{Float64}}
    _rewards::Array{Float64}
    discount_factor::Float64
end

function TestMDP(shape=(6,), stack=4, max_time=6, discount_factor=0.99)
    bad_state =  rand(1:50, shape...)
    normal_state =  rand(100:150, shape...)
    good_state =  rand(200:250, shape...)
    _observation_space = [bad_state, normal_state, good_state]
    _rewards = [-0.1, 0.0, 0.1]
    return TestMDP(shape, 4, stack, max_time, bad_state, normal_state, good_state, _observation_space, _rewards, discount_factor)
end

POMDPs.discount(mdp::TestMDP) = mdp.discount_factor

function POMDPs.observations(mdp::TestMDP)
    return mdp._observation_space
end

function POMDPs.actions(mdp::TestMDP)
    return 1:4
end

POMDPs.actionindex(mdp::TestMDP, a::Int64) = a

# s2o(s::Int64, pomdp::TestPOMDP) = observations(pomdp)[s]

function POMDPs.initialstate(mdp::TestMDP, rng::AbstractRNG)
    init_t = 1
    init_s = fill(1, mdp.stack)
    return (init_s, init_t)
end

function POMDPs.convert_s(t::Type{Vector{Float64}},s::Tuple{Vector{Int64}, Int64}, mdp::TestMDP)
    obs = zeros(mdp.shape..., mdp.o_stack)
    for i=1:mdp.o_stack
        obs[Base.setindex(axes(obs), i, ndims(obs))...] = observations(mdp)[s[1][end-i+1]]
    end
    return obs./255.0
end

function was_in_second(s::Tuple{Vector{Int64}, Int64})
    s[1][end] == 2
end

function POMDPs.generate_s(mdp::TestMDP, s::Tuple{Vector{Int64}, Int64}, a::Int64, rng::AbstractRNG)
    t_new = s[2] + 1 # increment time
    s_new = circshift(s[1], -1)
    if a < 4
        s_new[end] = a
    else
        s_new[end] = s_new[end-1]
    end
    return (s_new, t_new)
end

function POMDPs.reward(mdp::TestMDP, s::Tuple{Vector{Int64}, Int64}, a::Int64, sp::Tuple{Vector{Int64}, Int64})
    r = mdp._rewards[sp[1][end]]
    if was_in_second(s)
        r *= -10
    end
    return r
end

function POMDPs.isterminal(mdp::TestMDP, s)
    return s[2] >= mdp.max_time
end

function POMDPs.n_actions(mdp::TestMDP)
    return 4
end