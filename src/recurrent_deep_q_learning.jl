
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

## Fields:
- `arch::RecurrentQNetworkArchitecture` Specify the architecture of the Q network default = QNetworkArchitecture(conv=[], fc=[])
- `lr::Float64` learning rate default = 0.005
- `max_steps::Int64` total number of training step default = 1000
- `target_update_freq::Int64` frequency at which the target network is updated default = 500
- `batch_size::Int64` batch size sampled from the replay buffer default = 32
- `trace_length::Int64` trajectory length used to train the LSTM
- `train_freq::Int64` frequency at which the active network is updated default  = 4
- `log_freq::Int64` frequency at which to logg info default = 100
- `eval_freq::Int64` frequency at which to eval the network default = 100
- `num_ep_eval::Int64` number of episodes to evaluate the policy default = 100
- `eps_fraction::Float64` fraction of the training set used to explore default = 0.5
- `eps_end::Float64` value of epsilon at the end of the exploration phase default = 0.01
- `double_q::Bool` double q learning udpate default = true
- `dueling::Bool` dueling structure for the q network default = true
- `prioritized_replay::Bool` enable prioritized experience replay default = true
- `prioritized_replay_alpha::Float64` default = 0.6
- `prioritized_replay_epsilon::Float64` default = 1e-6
- `prioritized_replay_beta::Float64` default = 0.4
- `buffer_size::Int64` size of the experience replay buffer default = 1000
- `max_episode_length::Int64` maximum length of a training episode default = 100
- `train_start::Int64` number of steps used to fill in the replay buffer initially default = 200
- `grad_clip::Bool` enables gradient clipping default = true
- `clip_val::Float64` maximum value for the grad norm default = 10.0
- `rng::AbstractRNG` random number generator default = MersenneTwister(0)
- `verbose::Bool` default = true
"""
@with_kw mutable struct DeepRecurrentQLearningSolver
    arch::RecurrentQNetworkArchitecture = RecurrentQNetworkArchitecture()
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
