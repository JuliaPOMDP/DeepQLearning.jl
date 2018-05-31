@POMDP_require solve(solver::DeepQLearningSolver, problem::Union{MDP,POMDP}) begin
    P = typeof(problem)
    S = state_type(problem)
    A = action_type(problem)
    @req n_actions(::P)
    @req initial_state(::P, ::AbstractRNG)
    @req convert_o(::Type{Vector{Float64}}, ::S, ::P)
    @req action_index(::P, ::A)
    @req actions(::P)
    @req discount(::P)

end
