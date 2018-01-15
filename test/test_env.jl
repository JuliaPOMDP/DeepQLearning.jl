# ### Test environment
using POMDPs, POMDPToolbox, DeepRL

mutable struct TestPOMDP <: POMDP{Tuple{Vector{Int64}, Int64}, Int64, Array{Float64}}
    shape
    stack::Int64
    max_time::Int64
    bad_state::Array{Int64}
    normal_state::Array{Int64}
    good_state::Array{Int64}
    _observation_space::Array{Array{Float64}}
    _rewards::Array{Float64}
    discount_factor::Float64
end

function TestPOMDP(shape=(6,), stack=4, max_time=6, discount_factor=0.99)
    bad_state =  rand(1:50, shape...)
    normal_state =  rand(100:150, shape...)
    good_state =  rand(200:250, shape...)
    _observation_space = [bad_state, normal_state, good_state]
    _rewards = [-0.1, 0.0, 0.1]
    return TestPOMDP(shape, stack, max_time, bad_state, normal_state, good_state, _observation_space, _rewards, discount_factor)
end

POMDPs.discount(pomdp::TestPOMDP) = pomdp.discount_factor

function POMDPs.observations(pomdp::TestPOMDP)
    return pomdp._observation_space
end

function POMDPs.actions(pomdp::TestPOMDP)
    return 1:4
end

# s2o(s::Int64, pomdp::TestPOMDP) = observations(pomdp)[s]

function POMDPs.initial_state(pomdp::TestPOMDP, rng::AbstractRNG)
    init_t = 1
    init_s = fill(1, pomdp.stack)
    return (init_s, init_t)
end

function POMDPs.generate_o(pomdp::TestPOMDP, s::Tuple{Vector{Int64}, Int64}, rng::AbstractRNG)
    obs = zeros(pomdp.shape..., pomdp.stack)
    for i=1:pomdp.stack
        obs[Base.setindex(indices(obs), i, ndims(obs))...] = observations(pomdp)[s[1][i]]
    end
    return obs
end

function POMDPs.convert_o(t::Type{Vector{Float64}}, o::Array{Float64}, pomdp::TestPOMDP)
    return o./255.0
end

function was_in_second(s::Tuple{Vector{Int64}, Int64})
    s[1][end] == 2
end

function POMDPs.generate_s(pomdp::TestPOMDP, s::Tuple{Vector{Int64}, Int64}, a::Int64, rng::AbstractRNG)
    t_new = s[2] + 1 # increment time
    s_new = circshift(s[1], -1)
    if a < 4
        s_new[end] = a
    else
        s_new[end] = s_new[end-1]
    end
    return (s_new, t_new)
end

function POMDPs.generate_o(pomdp::TestPOMDP, s::Tuple{Vector{Int64}, Int64}, a::Int64, sp::Tuple{Vector{Int64}, Int64}, rng::AbstractRNG)
    return generate_o(pomdp, sp, rng)
end

function POMDPs.reward(pomdp::TestPOMDP, s::Tuple{Vector{Int64}, Int64}, a::Int64, sp::Tuple{Vector{Int64}, Int64})
    r = pomdp._rewards[sp[1][end]]
    if was_in_second(s)
        r *= -10
    end
    return r
end

function POMDPs.isterminal(pomdp::TestPOMDP, s)
    return s[2] >= pomdp.max_time
end

function POMDPs.n_actions(pomdp::TestPOMDP)
    return 4
end

rng = MersenneTwister(0)

env = POMDPEnvironment(TestPOMDP(1,))



for i=1:100
    nsteps = 10
    done = false
    r_tot = 0.0
    step = 0


    obs = reset(env)
    # println("start at t=0 obs $obs")
    # println("Start state $(env.state)")
    while !done && step <= nsteps
        a = sample_action(env)
        obs, rew, done, info = step!(env, a)
        # println("state ", env.state, " action ", a)
        # println("Reward ", rew)
        # println(obs, " ", done, " ", info, " ", step)
        r_tot += rew
        step += 1
    end

    println(r_tot)

end
