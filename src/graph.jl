### For building the tensorflow graph

const Q_SCOPE = "active_q"
const TARGET_Q_SCOPE = "target_q"

"""
Reset the graph and open a new session
"""
function init_session()
    g = Graph()
    tf.set_def_graph(g)
    sess = Session(g)
    return sess
end

"""
Create placeholders for DQN training: s, a, sp, r, done
The shape is inferred from the environment
"""
function build_placeholders(env::Union{POMDPEnvironment, MDPEnvironment})
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    @tf begin
        s = placeholder(Float32, shape=[-1, obs_dim...])
        a = placeholder(Int32, shape=[-1])
        sp = placeholder(Float32, shape=[-1, obs_dim...])
        r = placeholder(Float32, shape=[-1])
        done_mask = placeholder(Bool, shape=[-1])
    end
    return s, a, sp, r, done_mask
end


"""
Compute the Huber Loss
"""
function huber_loss(x, δ::Float64=1.0)
    mask = abs(x) .< δ
    return mask.*0.5.*x.^2 + (1-mask).*δ.*(abs(x) - 0.5*δ)
end

"""
Build the loss operation
relies on the Bellman equation
"""
function build_loss(env::Union{POMDPEnvironment, MDPEnvironment}, q::Tensor, target_q::Tensor, a::Tensor, r::Tensor, done_mask::Tensor)
    loss, td_errors = nothing, nothing
    variable_scope("loss") do
        term = cast(done_mask, Float32)
        A = one_hot(a, n_actions(env))
        q_sa = sum(A.*q, 2)
        q_samp = r + (1 - term).*discount(env.problem).*maximum(target_q, 2)
        td_errors = q_sa - q_samp
        errors = huber_loss(td_errors)
        loss = mean(errors)
    end
    return loss, td_errors
end

"""
Build train operation
Support gradient clipping
"""
function build_train_op(loss::Tensor;
                        lr::Union{Float64, Tensor} = 0.1,
                        grad_clip::Bool = true,
                        clip_val::Float64 = 10.,
                        optimizer_type=train.AdamOptimizer)
    optimizer = optimizer_type(lr)

    train_var = get_train_vars_by_name("active_q")

    grad_vars = train.compute_gradients(optimizer, loss, train_var)
    clip_grads = grad_vars
    if grad_clip
        clip_grads = [(clip_by_norm(gradvar[1], clip_val), gradvar[2]) for gradvar in grad_vars]
    end
    train_op = train.apply_gradients(optimizer, clip_grads)
    grad_norm = global_norm([g[1] for g in clip_grads])
    return train_op, grad_norm
end

"""
returns a tensorflow operation to update a target network
if you run the operation, it will copy the value of the weights and biases in q_scope to
the weights and biases in target_q_scope
"""
function build_update_target_op(q_scope=Q_SCOPE, target_q_scope=TARGET_Q_SCOPE)
    q_weights = get_train_vars_by_name(q_scope)
    target_q_weights = get_train_vars_by_name(target_q_scope)

    all_ops = [tf.assign(target_q_weights[i], q_weights[i]) for i in 1:length(q_weights)]
    return update_target_op = tf.group(all_ops..., name="update_target_op")
end


mutable struct TrainGraph
    sess::Session
    s::Tensor
    a::Tensor
    sp::Tensor
    r::Tensor
    done_mask::Tensor
    q::Tensor
    target_q::Tensor
    loss::Tensor
    td_errors::Tensor
    train_op::Tensor
    grad_norm::Tensor
    update_op::Tensor
end

function build_graph(solver::DeepQLearningSolver, env::Union{MDPEnvironment, POMDPEnvironment})
    sess = init_session()
    s, a, sp, r, done_mask = build_placeholders(env)
    q = build_q(s, solver.arch, env, Q_SCOPE)
    target_q = build_q(sp, solver.arch, env, TARGET_Q_SCOPE)
    loss, td_errors = build_loss(env, q, target_q, a, r, done_mask)
    train_op, grad_norm = build_train_op(loss,
                                         lr=solver.lr,
                                         grad_clip=solver.grad_clip,
                                         clip_val=solver.clip_val)
    update_op = build_update_target_op("active_q", "target_q")
    return TrainGraph(sess, s, a, sp, r, done_mask, q, target_q, loss, td_errors, train_op, grad_norm, update_op)
end
