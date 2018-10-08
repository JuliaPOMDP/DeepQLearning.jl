"""
    flatten_batch(x::AbstractArray)
flatten a multi dimensional array to keep only the last dimension.
It returns a 2 dimensional array of size (flatten_dim, batch_size)
"""
function flatten_batch(x::AbstractArray)
    reshape(x, (:, size(x)[end]))
end

"""
    huber_loss(x, δ::Float64=1.0)
Compute the Huber Loss
"""
function huber_loss(x, δ::Float64=1.0)
    if abs(x) < δ
        return 0.5*x^2
    else
        return δ*(abs(x) - 0.5*δ)
    end
end

"""
    is_recurrent(m)
returns true if m contains a recurrent layer 
"""
function is_recurrent(m)
    for layer in m 
        if layer isa Flux.Recur 
            return true 
        end
    end
end
