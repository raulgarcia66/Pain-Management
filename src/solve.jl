"""
Algorithm implementations for policy computation.
"""

using LinearAlgebra
using Pipe

function compute_policy(states, actions, action_sets, R, P, T; u_terminal=zeros(length(states)))
    # Compute policy via value iteration
    num_states = length(states)

    π = [zeros(Int, num_states) for _ in 1:T]   # policy via action indices
    u = [zeros(num_states) for _ in 1:T+1]
    u[8] = u_terminal

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

function compute_policy_time(states, actions, action_sets, R, P, T; u_terminal=zeros(length(states)))
    # Compute policy via value iteration
    num_states = length(states)

    π = [zeros(Int, num_states) for _ in 1:T]   # policy via action indices
    u = [zeros(num_states) for _ in 1:T+1]
    u[T+1] = u_terminal

    reachable_states = compute_reachable_states_vector(states, actions, P)
    # reachable_states[a][i] is indices of states reachable to i after taking action a
    for t = T:-1:1
        for i in eachindex(states)

            u_temp = map(a -> R[t,i,a] + sum([ P[a][i,j] * u[t+1][j] for j in reachable_states[a][i] ]), action_sets[i])
            u[t][i] = maximum(u_temp)

            π[t][i] = @pipe findfirst(==(u[t][i]), u_temp) |> action_sets[i][_]

        end
    end

    return u, π
end

function compute_policy_mult_actions(states, actions, action_sets::Vector{W}, R, P, T; u_terminal=zeros(length(states))) where {W}
    # Compute policy via value iteration
    num_states = length(states)

    # Policy via action indices
    if W <: Tuple
        π = [[(0,) for _ in 1:num_states] for _ in 1:T]
    else
        π = [[[0] for _ in 1:num_states] for _ in 1:T]
    end
    u = [zeros(num_states) for _ in 1:T+1]
    u[8] = u_terminal

    reachable_states = compute_reachable_states_vector(states, actions, P)
    # reachable_states[a][i] is indices of states reachable to i after taking action a
    for t = T:-1:1
        for i in eachindex(states)

            u_temp = map(a -> sum([ P[a][i,j] * (R[a,i,j] + u[t+1][j]) for j in reachable_states[a][i] ]), action_sets[i])
            u[t][i] = maximum(u_temp)

            π[t][i] = @pipe findall(==(u[t][i]), u_temp) |> action_sets[i][_]

        end
    end

    optimal_actions = Tuple{Int, Int}[]
    for t = 1:T, i = 1:num_states
        if length(π[t][i]) > 1
            push!(optimal_actions, (t,i))
            println("Multiple optical actions for (t,i) = ($t,$i)")
        end
    end

    return u, π, optimal_actions
end