using Distributions
using TensorFlow
const tf = TensorFlow
using Gadfly



# Generate some synthetic data
n_ex = 500

r1, r2 = 0.3, 0.7
sig = 0.05
t = linspace(0,2*pi, n_ex)
y = zeros(2*n_ex, 1)
x1 = zeros(n_ex, 2)
x1[:, 1] = r1*cos(t) + sig*randn(n_ex)
x1[:, 2] = r1*sin(t) + sig*randn(n_ex)
y[1:n_ex] = 1
x2 = zeros(n_ex, 2)
x2[:, 1] = r2*cos(t) + sig*randn(n_ex)
x2[:, 2] = r2*sin(t) + sig*randn(n_ex)
x = vcat(x1, x2)
p = randperm(2*n_ex)
x = x[p, :]
y = y[p]
y = reshape(y, (2*n_ex,1))
# x = rand(n_ex, 2)

# y = 1.0*((x[:,1] - 0.5).^2 + (x[:, 2] - 0.5).^2 .> 0.15 )
# y = reshape(y, (n_ex, 1))
plot(x=x[:,1], y=x[:,2], color=y)


# Build the model
g = Graph()
tf.set_def_graph(Graph())
sess = Session(g)



function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end


function bias_variable(shape)
    initial = zeros(shape...)
    return Variable(initial)
end

function dense(input::Tensor, hidden_units, activation=identity)
    input_shape = get(get_shape(input).dims[end])
    weight_shape = (input_shape, hidden_units)
    var = 1/input_shape # Xavier initialization
    fc_w = Variable(rand(Normal(0., var), weight_shape...))
    fc_b = Variable(zeros(hidden_units))
    a = activation(input*fc_w + fc_b)
    return a
end

function MLP(hiddens::Vector{Int64}, input::Tensor, num_output::Int64)
    a = input
    for h in hiddens
        a = dense(a, h, nn.relu)
    end
    a = dense(a, num_output, nn.sigmoid)
    return a
end


X = placeholder(Float64, shape=[-1, 2])
Y_obs = placeholder(Float64, shape=[-1, 1])

hiddens = [4,4]
out = MLP(hiddens, X, 1)

# a1 = dense(X, 4, nn.relu)
# a2 = dense(a1, 4, nn.relu)
# out = dense(a2, 1, nn.sigmoid)

# @tf begin
#     X = placeholder(Float64, shape=[-1, 2])
#     Y_obs = placeholder(Float64, shape=[-1, 1])
#     W1 = weight_variable((2, 4))
#     b1 = bias_variable((4,))
#     a1 = nn.relu(X*W1 + b1)
#     W = weight_variable((4, 1))
#     b = bias_variable((1,))
#     out = nn.sigmoid(a1*W + b)
# end

g.collections

#
# X = placeholder(Float64, shape=[-1, 2])
# Y_obs = placeholder(Float64, shape=[-1, 1])
#
# variable_scope("logisitic_model"; initializer=Normal(0, .001)) do
#     global W = get_variable("W", [2, 4], Float64)
#     global B = get_variable("B", [4], Float64)
#     global a = nn.relu(X*W + B)
#     global W1 = get_variable("W1", [4, 1], Float64)
#     global B1 = get_variable("B1", [1], Float64)
# end

# out=1/(1 + exp(-(a*W1 + B1)))

Loss = -mean(log(out).*Y_obs + log(1.0 - out).*(1.0 - Y_obs))
optimizer = train.AdamOptimizer(0.01)
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()
# Run training
run(sess, global_variables_initializer())

checkpoint_path = mktempdir()
info("Checkpoint files saved in $checkpoint_path")
outval = nothing
loss_hist = Float64[]
for epoch in 1:300
    cur_loss, outval, _ = run(sess, [Loss,out, minimize_op], Dict(X=>x, Y_obs=>y))
    push!(loss_hist, cur_loss)
    println(@sprintf("Current loss is %.2f.", cur_loss))
    train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)
end

loss_hist[end]
plot(x=1:length(loss_hist), y=loss_hist)

X_bound = rand(-1:0.01:1, 10000, 2)

y_pred = run(sess, out, Dict(X=>X_bound))
pred = 1*(y_pred.>0.5)
train_pred = 1*(outval.>0.5)
plot(x=x[:,1], y=x[:,2], color=train_pred)
plot(x=X_bound[:,1], y=X_bound[:,2], color=pred)

# visualize()
