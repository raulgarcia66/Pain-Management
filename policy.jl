"""
Define rewards and compute respective policies.
"""

include("./parameters.jl")

##### Rewards
# Action penalty: [+2, +1, 0, -1, -2]

# Patient functionality: [Poor, Acceptable, Good]
functionality_values = [-1,0,1]

# Patient pain level: [MM, MS]
pain_level_values = [0.5, 0]

reachable_states = compute_reachable_states_vector(states, actions, P)
# reachable_states[a][i] is indices of states reachable to i after taking action a
reachable_states[1][5]

states_with_poor = [i for i = 1:3:18]
states_with_acceptable = [i for i = 2:3:18]
states_with_good = [i for i = 3:3:18]
state_health_partition = [states_with_poor, states_with_acceptable, states_with_good]

states_with_MM = @pipe Iterators.flatten([[i; i+1; i+2] for i in 1:6:18]) |> collect(_)
states_with_MS = @pipe Iterators.flatten([[i; i+1; i+2] for i in 4:6:18]) |> collect(_)
state_pain_partition = [states_with_MM, states_with_MS]

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
            reward += functionality_values[health_list_ind_j] - functionality_values[health_list_ind_i]
            reward += pain_level_values[pain_list_ind_j] - pain_level_values[pain_list_ind_i]

            # TODO: Penalties
            R[a,i,j] = reward
        end
    end
end
R


##### Compute policy
term_reward_func = [-0.5; 0.75; 1]   # [Poor, Acceptable, Good]
term_reward_pain_sev = [0.5; -0.5]   # [MM, MS]
u_terminal = map(i -> begin
    local u = 0
    if i in states_with_good
        u += term_reward_func[3]
    elseif i in states_with_acceptable
        u += term_reward_func[2]
    elseif i in states_with_poor
        u += term_reward_func[1]
    end
    if i in states_with_MM
        u += term_reward_pain_sev[1]
    elseif i in states_with_MS
        u += term_reward_pain_sev[2]
    end
    return u
end, eachindex(states))

u, π = compute_policy(states, actions, action_sets, R, P, T; u_terminal = u_terminal)
u_mult, π_mult = compute_policy_mult_actions(states, actions, action_sets, R, P, T; u_terminal = zeros(num_states))

u[1]
π[1][1]

### Logging

exp_num = 3

filename_exp = "Experiment summary.txt"
fe = open(filename_exp, "a")

filename_policy = "Policy $exp_num.txt"
f = open(filename_policy, "w")

# Parameter info
write(fe, "Experiment $exp_num\n")
write(fe, "u_terminal = $u_terminal\n")
write(fe, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
write(fe, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
write(fe, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
write(fe, "pain_level_values = $pain_level_values   # MM, MS\n")
write(fe, "\n")
close(fe)

write(fe, "Experiment $exp_num\n")
write(fe, "u_terminal = $u_terminal\n")
write(fe, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
write(fe, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
write(fe, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
write(fe, "pain_level_values = $pain_level_values   # MM, MS\n")
write(fe, "\n")
write(f, "\n")
# Policy
write(f, "\tT\t\t")
for t = 1:T
    write(f, "$t\t\t")
end
write(f, "\n")
for i in eachindex(states)
    write(f, "$(states[i])\t")
    for t in 1:T
        write(f, "$(actions[π[t][i]])\t\t")
    end
    write(f, "\n")
end
close(f)