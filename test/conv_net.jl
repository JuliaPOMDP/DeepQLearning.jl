using Distributions
using TensorFlow
const tf = TensorFlow
using Gadfly

using MNIST

type DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float32, batch_size, 784)
    y = zeros(Float32, batch_size, 10)
    for i in 1:batch_size
        x[i, :] = trainfeatures(loader.order[loader.cur_id])
        label = trainlabel(loader.order[loader.cur_id])
        y[i, Int(label)+1] = 1.0
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x, y
end

function load_test_set(N=10000)
    x = zeros(Float32, N, 784)
    y = zeros(Float32, N, 10)
    for i in 1:N
        x[i, :] = testfeatures(i)
        label = testlabel(i)
        y[i, Int(label)+1] = 1.0
    end
    x,y
end


loader = DataLoader()



function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.1), shape...)
    return Variable(initial)
end

function conv2d(x, W)
    nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

function conv2d(inputs::Tensor, num_filters::Int64, kernel_size::Vector{Int64}, activation=identity; stride::Int64=1, padding::String="SAME")
    # assume inputs is of shape batch, height, width, channels
    input_channels = get(get_shape(inputs).dims[end])
    weight_shape = (kernel_size[1], kernel_size[2], input_channels, num_filters)
    var = 1/(kernel_size[1]*kernel_size[2]*input_channels) # Xavier initialization
    conv_W = Variable(map(Float32,rand(Normal(0., var), weight_shape...)))
    conv_b = Variable(zeros(num_filters))
    z = nn.conv2d(inputs, conv_W, Int64[1, stride, stride, 1], padding) + conv_b
    a = activation(z)
    a = cast(a, Float32)
    return a
end

function flatten(x::Tensor; batch_dim::Int64=1)
    shape = get_shape(x)
    N = length(shape.dims)
    flat_dim = compute_flat_dim(shape.dims)
    if batch_dim != 1 #permute dims such that it is in the first axis
        perm = [mod1(batch_dim + i, N) for i=0:N-1]
        x = permutedims(x, perm)
    end
    x = reshape(x, (-1, flat_dim))
    return x
end

"""
    This model takes as input an observation and returns values of all actions.
    Parameters
    ----------
    convs: [(int, (int,int), int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores
    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
"""

function cnn_to_mlp(inputs, convs, hiddens, num_output, final_activation=identity)
    out = inputs
    for (nfilters, kernel_size, stride) in convs
        out = conv2d(out, nfilters, kernel_size, nn.relu, stride=stride)
    end
    out = flatten(out)
    for h in hiddens
        out = dense(out, h, nn.relu)
    end
    out = dense(out, num_output, final_activation)
    return out
end


# Build the model
g = Graph()
tf.set_def_graph(Graph())
session = Session(g)

# @tf begin

x = placeholder(Float32)
y_ = placeholder(Float32, shape=[-1, 10])

x_image = reshape(x, [-1, 28, 28, 1])

y_conv = cnn_to_mlp(x_image, [(32, [5,5], 1), (64, [5,5], 1)], [1024], 10, nn.softmax)

# h_conv1 = conv2d(x_image, 32, [5,5], nn.relu)
# h_pool1 = max_pool_2x2(h_conv1)
#
# h_conv2 = conv2d(h_pool1, 64, [5,5], nn.relu)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # h_pool2_flat = reshape(h_pool2, [-1, 7*7*64])
# h_pool2_flat = flatten(h_pool2)
# h_fc1 = dense(h_pool2_flat, 1024, nn.relu)
#
#
# keep_prob = placeholder(Float32, shape=[])
# h_fc1_drop = nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = nn.softmax(h_fc1_drop * W_fc2 + b_fc2)

cross_entropy = reduce_mean(-reduce_sum(y_.*log(y_conv), axis=[2]))
# end




train_step = train.minimize(train.AdamOptimizer(1e-4), cross_entropy)

correct_prediction = indmax(y_conv, 2) .== indmax(y_, 2)

accuracy = reduce_mean(cast(correct_prediction, Float32))

run(session, global_variables_initializer())

loss_hist = Float64[]
for i in 1:200
    batch = next_batch(loader, 50)
    if i%100 == 1
        train_accuracy = run(session, accuracy, Dict(x=>batch[1], y_=>batch[2]))
        info("step $i, training accuracy $train_accuracy")
    end
    loss, _ = run(session, [cross_entropy,train_step], Dict(x=>batch[1], y_=>batch[2]))
    push!(loss_hist, loss)
end
loss_hist
loss_hist[1]
plot(x=1:length(loss_hist), y=loss_hist)

testx, testy = load_test_set()

println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))
