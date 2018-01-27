

"""
Build a q network given an architecture
"""
function build_q(input::Tensor,
                arch::QNetworkArchitecture,
                env::MDPEnvironment;
                scope::String="",
                reuse::Bool=false,
                dueling::Bool=false)
    return cnn_to_mlp(input, arch.conv, arch.fc, n_actions(env), scope=scope, reuse=reuse, dueling=dueling)
end
