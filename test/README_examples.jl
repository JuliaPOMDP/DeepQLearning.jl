using DeepQLearning
using POMDPs
using Flux
using POMDPModels
using POMDPTools

@testset "README Example 1" begin
    # load MDP model from POMDPModels or define your own!
    mdp = SimpleGridWorld();

    # Define the Q network (see Flux.jl documentation)
    # the gridworld state is represented by a 2 dimensional vector.
    model = Chain(Dense(2, 32), Dense(32, length(actions(mdp))))

    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2));

    solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                                exploration_policy = exploration,
                                learning_rate=0.005,log_freq=500,
                                recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
    policy = solve(solver, mdp)

    sim = RolloutSimulator(max_steps=30)
    r_tot = simulate(sim, mdp, policy)
    println("Total discounted reward for 1 simulation: $r_tot")
end

@testset "README Example 2" begin
    # Without using CuArrays
    mdp = SimpleGridWorld();

    # the model weights will be send to the gpu in the call to solve
    model = Chain(Dense(2, 32), Dense(32, length(actions(mdp))))

    exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2));
    
    solver = DeepQLearningSolver(qnetwork=model, max_steps=10000, 
                                exploration_policy=exploration,
                                learning_rate=0.005,log_freq=500,
                                recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
    policy = solve(solver, mdp)
end
