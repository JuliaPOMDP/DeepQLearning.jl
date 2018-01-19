
"""
Specify the Q network architecture
[(int, (int,int), int)]
"""
@with_kw mutable struct QNetworkArchitecture
    fc::Vector{Int64} = Vector{Int64}[]
    conv::Vector{Tuple{Int64, Vector{Int64}, Int64}} = Vector{Tuple{Int64, Tuple{Int64, Int64}, Int64}}[]
end

"""
Build a q network given an architecture
"""
function build_q(input::Tensor, arch::QNetworkArchitecture, env::Union{POMDPEnvironment, MDPEnvironment}, scope::String)
    return cnn_to_mlp(input, arch.conv, arch.fc, n_actions(env), scope=scope)
end
