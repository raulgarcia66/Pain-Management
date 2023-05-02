using LinearAlgebra
using Pipe

function compute_policy(states, actions, action_sets, R, P, T)
    # Compute policy via value iteration
    num_states = length(states)
    # num_actions = length(actions)

    π = [zeros(Int, num_states) for _ in 1:T]   # policy via action indices
    u = [zeros(num_states) for _ in 1:T+1]
    # u_temp = zeros(num_states)

    reachable_states = compute_reachable_states_vector(states, actions, P)
    # reachable_states[a][i] is indices of states reachable to i after taking action a
    for t = T:-1:1
        for i in eachindex(states)

            u_temp = map(a -> sum([ P[a][i,j] * (R[a,i,j] + u[t+1][j]) for j in reachable_states[a][i] ]), action_sets[i])
            u[t][i] = maximum(u_temp)

            π[t][i] = @pipe findfirst(==(u[t][i]), u_temp) |> action_sets[i][_]

        end
    end

    return u, π
end
