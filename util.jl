
function compute_reachable_states(current_state, action, P)
    # Returns indices of reachable states
    ϵ = 1E-5
    return filter(j -> P[action][current_state, j] > ϵ, 1:size(P[action],2))
end


function compute_reachable_states_vector(states, actions, P)
    # Returns vector X, where X[a][i] is the indices of states reachable to i after taking action a
    return map(a -> map(i ->  compute_reachable_states(i, a, P), eachindex(states)), eachindex(actions))
end


function find_list_index(state, lists)
    # index = 0
    for k = eachindex(lists)
        if state in lists[k]
            return k
            # index = k
            # break
        end
    end
    error("Current state $i not found in state_health_partition")
    # if index == 0
        # error("Current state $i not found in state_health_partition")
    # end
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
    error("No state generated. Check transition probability matrix.")
end