# Highly inspired from https://github.com/tejank10/Flux-baselines/blob/master/baselines/dqn/duel-dqn.jl
struct DuelingNetwork
    base::Chain
    val::Chain
    adv::Chain
end

function (m::DuelingNetwork)(inpt)
    x = m.base(inpt)
    return m.val(x) .+ m.adv(x) .- mean(m.adv(x), dims=1)
end

Flux.@layer DuelingNetwork

function Flux.reset!(m::DuelingNetwork)
    Flux.reset!(m.base)
end

function Base.iterate(m::DuelingNetwork, i=1)
    if i > length(m.base.layers) + length(m.val.layers) + length(m.adv.layers)
        return nothing 
    end
    if i <= length(m.base.layers)
        return (m.base[i], i+1)
    elseif i <= length(m.base.layers) + length(m.val.layers)
        return (m.val[i - length(m.base.layers)], i+1)
    elseif i <= length(m.base.layers) + length(m.val.layers) + length(m.adv.layers)
        return (m.adv[i - length(m.base.layers) - length(m.val.layers)], i+1)
    end   
end

function Base.deepcopy(m::DuelingNetwork)
  DuelingNetwork(deepcopy(m.base), deepcopy(m.val), deepcopy(m.adv))
end

function create_dueling_network(m::Chain)
    duel_layer = -1
    for i=1:length(m.layers)
        l = m[end-i+1]
        if !isa(l, Dense)
            duel_layer = length(m.layers)-i+1
            break
        elseif i == length(m.layers)
            duel_layer = 0
        end
    end
    error_str = "DeepQLearningError: the qnetwork provided is incompatible with dueling"
    duel_layer == -1 ? throw(error_str) : nothing
    for l in m[duel_layer+1:end]
        @assert isa(l, Dense) error_str
    end
    nlayers = length(m.layers)
    _, last_layer_size = size(m[end].weight)
    val = Chain([deepcopy(m[i]) for i=duel_layer+1:nlayers-1]..., Dense(last_layer_size, 1))
    adv = Chain([deepcopy(m[i]) for i=duel_layer+1:nlayers]...)
    base = Chain(identity, [deepcopy(m[i]) for i=1:duel_layer+1-1]...)
    return DuelingNetwork(base, val, adv)
end
