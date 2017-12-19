using Base.Test
#TODO test fatten

function test_flatten_multi_dims(x::Array{Float64})
    for dim=1:3
        x_ = flatten(x, batch_dim=dim)
        bs = size(x)[dim]
        for i=1:bs
            if !(sum(slicedim(x, dim, i)[:]) ≈ sum(x_[i,:]))
                return false
            end
        end
    end
    return true
end

x = randn(100,200)
@test flatten(x) == x
@test flatten(x, batch_dim=2) == transpose(x)

x1 = randn(10,20,30)
x2 = randn(10,20,30,40)
x3 = randn(rand(1:100),rand(1:100),rand(1:100),rand(1:100))
@test test_flatten_multi_dims(x1)
@test test_flatten_multi_dims(x2)
@test test_flatten_multi_dims(x3)
