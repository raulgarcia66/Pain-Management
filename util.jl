"""
Helper/utility functions.
"""

function compute_reachable_states(current_state::Int, action::Int, P)
    # Returns indices of reachable states
    ϵ = 1E-5
    return filter(j -> P[action][current_state, j] > ϵ, 1:size(P[action],2))
end


function compute_reachable_states_vector(states, actions, P)
    # Returns vector X, where X[a][i] is the indices of states reachable to i after taking action a
    return map(a -> map(i ->  compute_reachable_states(i, a, P), eachindex(states)), eachindex(actions))
end


function gen_next_state(current_state::Int, action::Int, P)
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

##############################################################################

function find_list_index(state::Int, lists::Vector{Vector{Int}})
    for k = eachindex(lists)
        if state in lists[k]
            return k
        end
    end
    error("Current state $i not found in state_health_partition")
end

"""
Rewards are received for when a transition from one state to another occurs.
"""
function construct_R_transition(states, actions, action_sets, P, α, state_health_partition, state_pain_partition, 
                    functionality_values, pain_level_values, action_penalty)
    num_states = length(states); num_actions = length(actions);
    reachable_states = compute_reachable_states_vector(states, actions, P)   # reachable_states[a][i] is indices of states reachable to i after taking action a
    R = zeros(num_actions, num_states, num_states)
    
    for i in eachindex(states)
        for a in action_sets[i]
            # println("In state $i, taking action $(actions[a]), able to reach states $(reachable_states[a][i])\n")
            for j in reachable_states[a][i]

                ### Current state i
                health_list_ind_i = find_list_index(i, state_health_partition)
                pain_list_ind_i = find_list_index(i, state_pain_partition)
                ### Future state j
                health_list_ind_j = find_list_index(j, state_health_partition)
                pain_list_ind_j = find_list_index(j, state_pain_partition)

                reward = 0
                reward += action_penalty(a, α)
                reward += functionality_values[health_list_ind_j] - functionality_values[health_list_ind_i]
                reward += pain_level_values[pain_list_ind_j] - pain_level_values[pain_list_ind_i]

                R[a,i,j] = reward
            end
        end
    end
    
    return R
end


"""
Rewards are received every period depending on the state one occupies.
"""
function construct_R_state_occupation(states, actions, action_sets, P, α, state_health_partition, state_pain_partition, 
    functionality_values, pain_level_values, action_penalty)
    num_states = length(states); num_actions = length(actions);
    reachable_states = compute_reachable_states_vector(states, actions, P)   # reachable_states[a][i] is indices of states reachable to i after taking action a
    R = zeros(num_actions, num_states, num_states)

    for i in eachindex(states)
        for a in action_sets[i]
            # println("In state $i, taking action $(actions[a]), able to reach states $(reachable_states[a][i])\n")
            for j in reachable_states[a][i]

            ### Current state i
            # health_list_ind_i = find_list_index(i, state_health_partition)
            # pain_list_ind_i = find_list_index(i, state_pain_partition)
            ### Future state j
            health_list_ind_j = find_list_index(j, state_health_partition)
            pain_list_ind_j = find_list_index(j, state_pain_partition)

            reward = 0
            reward += action_penalty(a, α)
            reward += functionality_values[health_list_ind_j]   # - functionality_values[health_list_ind_i]
            reward += pain_level_values[pain_list_ind_j]   # - pain_level_values[pain_list_ind_i]

            R[a,i,j] = reward
            end
        end
    end

    return R
end