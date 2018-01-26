
# include experience replay

pomdp = TestPOMDP((5,5), 1, 6)
env = POMDPEnvironment(pomdp)

solver = DeepQLearningSolver(max_steps=10000, lr=0.001, eval_freq=1000,num_ep_eval=100,
                            arch = QNetworkArchitecture(conv=[], fc=[64,32]),
                            double_q = false, dueling=true)
trace_length = 8 #hp to add to solver

arch = RecurrentQNetworkArchitecture()

sess = init_session()

# placeholder
obs_dim = obs_dimensions(env)
n_outs = n_actions(env)
# bs x trace_length x dim

s = placeholder(Float32, shape=[-1,-1, obs_dim...])
a = placeholder(Int32, shape=[-1, -1])
sp = placeholder(Float32, shape=[-1,-1, obs_dim...])
r = placeholder(Float32, shape=[-1, -1])
done_mask = placeholder(Bool, shape=[-1, -1])
trace_mask = placeholder(Int64, shape=[-1, -1])
w = placeholder(Float32, shape=[-1])

# q network
# inputs
inputs = s
convs = []
hiddens_in = []
hiddens_out = []
num_output = n_actions(env)
final_activation = identity
scope = "drqn"
reuse = false
dueling = false
# body of the function

# retrieve dynamic shapes
input_shape = tf.shape(inputs)
input_ndims = ndims(s)
obs_dim = [get(d) for d in get_shape(s).dims[3:end]]
input_dims = [input_shape[i] for i=1:input_ndims]

# assumes the first two dimensions are batch size and trace length
out = inputs
flatten_time_dims = vcat(input_dims[1].*input_dims[2], input_dims[3:end]...)

# out = permutedims(out, [2,1,[i for i=3:input_ndims]...]) # for reshape consistency
out = reshape(out, flatten_time_dims)
out = reshape(out, (-1, obs_dim...)) # should not do anything but get the tensor shape not unknown
# feed into conv_to_mlp
out = cnn_to_mlp(out, convs, hiddens_in, 0, scope=scope, reuse=reuse, dueling=false, final_activation=nn.relu)
#retrieve time dimension
flat_dim = get(get_shape(out).dims[end])
non_flat_time_dims = vcat(input_dims[1], input_dims[2], 25)
out = reshape(out, non_flat_time_dims)

# build RNN
rnn_cell = nn.rnn_cell.LSTMCell(arch.lstm_size)
c = placeholder(Float64, shape=[-1, arch.lstm_size])
h =  placeholder(Float64, shape=[-1, arch.lstm_size])
state_in = LSTMStateTuple(c, h)
last_out, state, out = dynamic_rnn(rnn_cell, out,initial_state=state_in,
                                input_dim = flat_dim, reuse=reuse)

# output with dueling
flatten_time_dims = vcat(input_dims[1].*input_dims[2], tf.shape(out)[end])
out = reshape(out, flatten_time_dims)
out = reshape(out, (-1, arch.lstm_size)) # should not do anything but get the tensor shape not unknown
out = cnn_to_mlp(out, [], arch.fc_out, n_actions(env), scope=scope, reuse=reuse, dueling=true )

s_val = rand(4,37,5,5,1)
feed_dict = Dict(s=>s_val, c=>zeros(4,64), h=>zeros(4,64))



run(sess, global_variables_initializer())
@time out_val, _ = run(sess, [out, state], feed_dict)

all_time_val = reshape(all_time_val, (2,3,64))

debug

mean(out_val)



s_ = reshape(s_val, (6,5,5,1))
s_ = reshape(s_, (6,25))

# test RNN
a = placeholder(Float32, shape=[-1,-1, 4, 6])
# build RNN
rnn_cell = nn.rnn_cell.LSTMCell(32)
c = placeholder(Float64, shape=[-1, arch.lstm_size])
h =  placeholder(Float64, shape=[-1, arch.lstm_size])
state_in = LSTMStateTuple(c, h)
out, state, all_time = dynamic_rnn(rnn_cell, out,initial_state=state_in,
                                input_dim = flat_dim, reuse=reuse)

# feed_dict
a_val = rand()










sess = init_session()

tl = placeholder(Int64, shape=[])

a = placeholder(Float32, shape=[-1,-1,5,5])
a_shape = tf.shape(a)

out = a
dim = vcat(a_shape[1].*a_shape[2], a_shape[3], a_shape[4])
out = reshape(out, dim)
out = 2*out
dim2 = vcat(a_shape[1], a_shape[2], a_shape[3], a_shape[4])
out = reshape(out, dim2)


a_val = rand(2,3,5,5)

# b, state = dynamic_rnn(nn.rnn_cell.LSTMCell(64), b)

run(sess, global_variables_initializer())
run(sess, out, Dict(a=>a_val))

function Base.ndims(A::AbstractTensor)
    length(get_shape(A).dims)
end
