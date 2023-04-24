
function compute_reachable_states(current_state, action, P)
    # Returns indices of reachable states
    ϵ = 1E-5
    return filter(j -> P[action][current_state, j] > ϵ, 1:size(P[action],2))
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