"""
Check parameters and rewards for validity and monotoneness.
"""

include("./parameters.jl")
include("./util.jl")
include("./solve.jl")

rows_dropped = "none"
method = "LR"
order = "ascending"

#### Rewards
filename_no_week_no_ext = "Rewards imputed $method order $order $rows_dropped dropped"
R = load_R(states, actions, action_sets, filename_no_week_no_ext, T)

#### Compute policy
u, π = compute_policy(states, actions, action_sets, R, P, T)
t = 1
i = 5
u[t][i]
π[t][i]
actions[π[t][i]]

policy_matrix = hcat(map(t -> actions[π[t]], 1:T)...)

##################################################################################
##################################################################################
# Monotoneness of rewards over time
T = 6
time_states = [Int[] for _ in eachindex(states), _ in eachindex(actions)]

for i in eachindex(states), a in action_sets[i]

    current_trend = ""
    if R[2][i,a] < R[1][i,a]
        current_trend = "nonincreasing"
    else
        current_trend = "nondecreasing"
    end

    monotone = true
    for t = 3:T
        if (R[t][i,a] < R[t-1][i,a] && current_trend == "nondecreasing") # ||
                        # (R[t][i,a] > R[t-1][i,a] && current_trend == "nonincreasing")
            monotone = false
            current_trend = "nonincreasing"
            push!(time_states[i,a], t)
        elseif (R[t][i,a] > R[t-1][i,a] && current_trend == "nonincreasing")
            monotone = false
            current_trend = "nondecreasing"
            push!(time_states[i,a], t)
        end
    end

end

collect(eachrow(time_states))

# Summary: Def not monotone. Oscillates.

##################################################################################
##################################################################################
# Monotoneness of value function
T = 6
time_states = [Int[] for _ in eachindex(states)]

for i in eachindex(states)

    current_trend = ""
    if u[2][i] < u[1][i]
        current_trend = "nonincreasing"
    else
        current_trend = "nondecreasing"
    end

    monotone = true
    for t = 3:T
        if (u[t][i] < u[t-1][i] && current_trend == "nondecreasing") # ||
                        # (u[t][i] > u[t-1][i] && current_trend == "nonincreasing")
            monotone = false
            current_trend = "nonincreasing"
            push!(time_states[i], t)
        elseif (u[t][i] > u[t-1][i] && current_trend == "nonincreasing")
            monotone = false
            current_trend = "nondecreasing"
            push!(time_states[i], t)
        end
    end

end

collect(eachrow(time_states))

# Summary: Not monotone

##################################################################################
##################################################################################
# Vary one state component, fix others. Check for one period at a time

