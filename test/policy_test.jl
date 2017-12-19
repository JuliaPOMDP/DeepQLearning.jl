using Distributions
using TensorFlow

# Generate some synthetic data
x = rand(100, 2)
y = 1.0*(x[:,1] .> x[:, 2].^2)

function draw(probs)
    y = zeros(size(probs))
    for i in 1:size(probs, 1)
        idx = rand(Categorical(probs[i, :]))
        y[i, idx] = 1
    end
    return y
end

# Build the model
sess = Session(Graph())
X = placeholder(Float64)
Y_obs = placeholder(Float64)

variable_scope("logistic_model_fc", initializer=Normal(0, .001)) do
    W = get_variable("weights", [2, 10], Float64)
    B = get_variable("bias", [10], Float64)
end
a1=nn.relu(X*W + B)
variable_scope("logistic_model_out", initializer=Normal(0, .001)) do
    W1 = get_variable("weights", [10, 1], Float64)
    B1 = get_variable("bias", [1], Float64)
end
out=nn.sigmoid(a1*W1 + B1)

Loss = -reduce_mean(reduce_sum(log(out).*Y_obs, axis=2))
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()
# Run training
run(sess, global_variables_initializer())
checkpoint_path = mktempdir()
info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:1000
    cur_loss, _ = run(sess, [Loss, minimize_op], Dict(X=>x, Y_obs=>y))
    println(@sprintf("Current loss is %.2f.", cur_loss))
    train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)
end
