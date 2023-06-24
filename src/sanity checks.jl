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
time_states[2,1]
# Summary: Def not monotone. Oscillates.

##################################################################################
##################################################################################
# Monotoneness of value function
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
# Vary one state component, fix others. Check value function for one period at a time

states = ["[0,P,MM]", "[0,A,MM]", "[0,G,MM]", "[0,P,MS]", "[0,A,MS]", "[0,G,MS]",
    "[1,P,MM]", "[1,A,MM]", "[1,G,MM]", "[1,P,MS]", "[1,A,MS]", "[1,G,MS]",
    "[2,P,MM]", "[2,A,MM]", "[2,G,MM]", "[2,P,MS]", "[2,A,MS]", "[2,G,MS]"]
num_states = length(states)

actions = ["+2", "+1", "0" , "-1", "-2"]
num_actions = length(actions)

action_sets = [(1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3),
              (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4),
              (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,5)]  # state 18 modified

#### Vary first component
linear_pain_med_vec = [[true for _ = 1:6] for _ = 1:T] # 6 states to check
initial_trend_pain_med_vec = [["" for _ = 1:6] for _ = 1:T]
for t = 1:T
    state_iter = collect(1:6)
    for s = eachindex(state_iter)

        set = [state_iter[s], state_iter[s]+6, state_iter[s]+12]
        if u[t][set[1]] > u[t][set[2]]
            # initial_trend = "decreasing"
            initial_trend_pain_med_vec[t][s] = "decreasing"
            if u[t][set[2]] >= u[t][set[3]]
                # linear_pain_med = true
                linear_pain_med_vec[t][s] = true
            else
                # linear_pain_med = false
                linear_pain_med_vec[t][s] = false
            end
        elseif u[t][set[1]] < u[t][set[2]]
            # initial_trend = "increasing"
            initial_trend_pain_med_vec[t][s] = "increasing"
            if u[t][set[2]] <= u[t][set[3]]
                # linear_pain_med = true
                linear_pain_med_vec[t][s] = true
            else
                # linear_pain_med = false
                linear_pain_med_vec[t][s] = true
            end
        end

    end
end

for time = 1:T, state_num = 1:6
    println("\n(time, state_num) = ($time, $state_num)")
    linear_pain_med_vec[time][state_num] ? println("Monotone") : println("Nonmonotone")
    initial_trend_pain_med_vec[time][state_num] == "increasing" ? println("Increasing initially") : println("Increasing initially")
end

#### Vary second component
linear_func_vec = [[true for _ = 1:6] for _ = 1:T] # 6 states to check
initial_trend_func_vec = [["" for _ = 1:6] for _ = 1:T]
for t = 1:T
    state_iter = [collect(1:3); collect(7:9); collect(13:15)]
    state_iter = collect(1:3:18)
    for s = eachindex(state_iter)

        set = [state_iter[s], state_iter[1]+1, state_iter[s]+2]
        if u[t][set[1]] > u[t][set[2]]
            # initial_trend = "decreasing"
            initial_trend_func_vec[t][s] = "decreasing"
            if u[t][set[2]] >= u[t][set[3]]
                # linear_pain_med = true
                linear_func_vec[t][s] = true
            else
                # linear_pain_med = false
                linear_func_vec[t][s] = false
            end
        elseif u[t][set[1]] < u[t][set[2]]
            # initial_trend = "increasing"
            initial_trend_func_vec[t][s] = "increasing"
            if u[t][set[2]] <= u[t][set[3]]
                # linear_pain_med = true
                linear_func_vec[t][s] = true
            else
                # linear_pain_med = false
                linear_func_vec[t][s] = true
            end
        end

    end
end

for time = 1:T, state_num = 1:6
    println("\n(time, state_num) = ($time, $state_num)")
    linear_func_vec[time][state_num] ? println("Monotone") : println("Nonmonotone")
    initial_trend_func_vec[time][state_num] == "increasing" ? println("Increasing initially") : println("Increasing initially")
end

#### Vary third component
linear_pain_level_vec = [[true for _ = 1:9] for _ = 1:T] # 9 states to check
initial_trend_pain_level_vec = [["" for _ = 1:9] for _ = 1:T]
for t = 1:T
    state_iter = [collect(1:3); collect(7:9); collect(13:15)]
    for s = eachindex(state_iter)

        set = [state_iter[s], state_iter[1]+3]
        if u[t][set[1]] > u[t][set[2]]
            # initial_trend = "decreasing"
            initial_trend_pain_level_vec[t][s] = "decreasing"
            # if u[t][set[2]] >= u[t][set[3]]
            #     # linear_pain_med = true
            #     linear_pain_level_vec[t][s] = true
            # else
            #     # linear_pain_med = false
            #     linear_pain_level_vec[t][s] = false
            # end
        elseif u[t][set[1]] < u[t][set[2]]
            # initial_trend = "increasing"
            initial_trend_pain_level_vec[t][s] = "increasing"
            # if u[t][set[2]] <= u[t][set[3]]
            #     # linear_pain_med = true
            #     linear_pain_level_vec[t][s] = true
            # else
            #     # linear_pain_med = false
            #     linear_pain_level_vec[t][s] = true
            # end
        end

    end
end

for time = 1:T, state_num = 1:9
    println("\n(time, state_num) = ($time, $state_num)")
#     linear_pain_level_vec[time][state_num] ? println("Monotone") : println("Nonmonotone")
    initial_trend_pain_level_vec[time][state_num] == "increasing" ? println("Increasing") : println("Decreasing")
end


# TODO: Do the state component as above over time as well