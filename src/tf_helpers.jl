###################################################################################################
# Helpers function for Tensorflow.jl to make building models easier
#

"""
    flatten(x::Array{Float64}; batch_dim::Int64=1)
flatten an array and keep the batch size

"""
function flatten(x::Array{Float64}; batch_dim::Int64=1)
    x_dims = size(x)
    N = ndims(x)
    batch_size = x_dims[batch_dim]
    if batch_dim != 1 #permute dims such that it is in the first axis
        perm = [mod1(batch_dim + i, N) for i=0:N-1]
        x = permutedims(x, perm)
    end
    flat_dim = div(length(x), batch_size)
    @assert batch_size == size(x)[1]
    x = reshape(x, (batch_size, flat_dim))
    return x
end

"""
define a fully connected layer
takes an inpt of shape (batch_size, dim) and outputs (batch_size, num_outputs)
"""
function dense(inpt, num_outputs, activation)
end
