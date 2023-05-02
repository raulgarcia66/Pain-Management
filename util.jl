
function compute_reachable_states(current_state, action, P)
    # Returns indices of reachable states
    ϵ = 1E-5
    return filter(j -> P[action][current_state, j] > ϵ, 1:size(P[action],2))
end


function compute_reachable_states_vector(states, actions, P)
    # Returns indices of reachable states
    return map(a -> map(i ->  compute_reachable_states(i, a, P), eachindex(states)), eachindex(actions))
end


function gen_next_state(current_state, action, P)
    U = rand()
    sum = 0
    for i in eachindex(P[action][current_state, :])
        sum += P[current_state, i, action]
        if sum > U
            return i
        end
    end
    error("No state generated.")
end