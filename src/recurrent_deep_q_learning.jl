
"""
    RecurrentQNetworkArchitecture
    specify an architecture with the following form:
    [CONV]-[FC]-[LSTM]-[FC]
"""
@with_kw mutable struct RecurrentQNetworkArchitecture
   fc_in::Vector{Int64} = Vector{Int64}[]
   convs::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
   fc_out::Vector{Int64} = Vector{Int64}[]
   lstm_size::Int64 = 32
end

"""
    DeepRecurrentQLearningSolver
Deep Q learning with a recurrent module to solve POMDPs
"""
@with_kw mutable struct DeepRecurrentQLearningSolver
    arch::QNetworkArchitecture = RecurrentQNetworkArchitecture()
    lr::Float64 = 0.001
    max_steps::Int64 = 1000
    target_update_freq::Int64 = 500
    batch_size::Int64 = 32
    trace_length = 6
    train_freq::Int64  = 4
    log_freq::Int64 = 100
    eval_freq::Int64 = 100
    num_ep_eval::Int64 = 100
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    double_q::Bool = true
    dueling::Bool = true
    buffer_size::Int64 = 10000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    grad_clip::Bool = true
    clip_val::Float64 = 10.0
    rng::AbstractRNG = MersenneTwister(0)
    verbose::Bool = true
end
