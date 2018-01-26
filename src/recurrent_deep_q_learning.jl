

"""
Specify the Q network architecture
[(int, (int,int), int)]
"""
@with_kw mutable struct RecurrentQNetworkArchitecture
   fc_in::Vector{Int64} = Vector{Int64}[]
   conv::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
   lstm_size::Int64 = 64
   fc_out::Vector{Int64} = Vector{Int64}[]
end

"""
    Deep Q Learning Solver type
"""
@with_kw mutable struct RecurrentDeepQLearningSolver
    arch::RecurrentQNetworkArchitecture = RecurrentQNetworkArchitecture(conv=[], fc=[])
    lr::Float64 = 0.005
    max_steps::Int64 = 1000
    target_update_freq::Int64 = 500
    batch_size::Int64 = 32
    trace_length::Int64 = 4 # what should the default be
    train_freq::Int64  = 4
    log_freq::Int64 = 100
    eval_freq::Int64 = 100
    num_ep_eval::Int64 = 100
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    double_q::Bool = true
    dueling::Bool = true
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    verbose::Bool = true
end
