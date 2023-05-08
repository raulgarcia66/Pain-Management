"""
Define rewards and compute respective policies.
"""

include("./parameters.jl")

##### Rewards
states_with_poor = [i for i = 1:3:18]
states_with_acceptable = [i for i = 2:3:18]
states_with_good = [i for i = 3:3:18]
state_health_partition = [states_with_poor, states_with_acceptable, states_with_good]

states_with_MM = @pipe Iterators.flatten([[i; i+1; i+2] for i in 1:6:18]) |> collect(_)
states_with_MS = @pipe Iterators.flatten([[i; i+1; i+2] for i in 4:6:18]) |> collect(_)
state_pain_partition = [states_with_MM, states_with_MS]

# Patient functionality: [Poor, Acceptable, Good]
functionality_values = [-1, 0, 1]

# Patient pain level: [MM, MS]
pain_level_values = [0.5, 0]

# Action penalties
action_penalty(action_index, α = 1) = -α * (action_index == 1 ? 1 : 0)   # Penalize taking action "+2"
α = 0.1

# Reward matrix
R = contruct_R(states, actions, action_sets, α, state_health_partition, state_pain_partition)


##### Compute policy
term_reward_func = [-0.5, 0.75, 1]   # [Poor, Acceptable, Good]
term_reward_pain_sev = [0.5, -0.5]   # [MM, MS]
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
u_mult, π_mult, opt_acts_mult = compute_policy_mult_actions(states, actions, action_sets, R, P, T; u_terminal = u_terminal)

u[1]
π[1][1]
π_mult[1][1]

### Logging

exp_count = 7

filename_exp = "Experiment summary.txt"
fe = open(filename_exp, "a")

filename_policy = "Policy $exp_count.txt"
f = open(filename_policy, "w")

# Parameter info
write(fe, "Experiment $exp_count\n")
write(fe, "u_terminal = $u_terminal\n")
write(fe, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
write(fe, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
write(fe, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
write(fe, "pain_level_values = $pain_level_values   # MM, MS\n")
write(fe, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
if length(opt_acts_mult) > 0
    write(fe, "Multiple optimal actions at: $opt_acts_mult\n")
end
write(fe, "Total expected reward u = $(u[1])\n")
write(fe, "\n")
close(fe)

write(f, "Experiment $exp_count\n")
write(f, "u_terminal = $u_terminal\n")
write(f, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
write(f, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
write(f, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
write(f, "pain_level_values = $pain_level_values   # MM, MS\n")
write(f, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
if length(opt_acts_mult) > 0
    write(fe, "Multiple optimal actions at: $opt_acts_mult\n")
end
write(f, "Total expected reward u = $(u[1])\n")
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
write(f, "\n")
close(f)

##############################################################################

# Patient functionality: [Poor, Acceptable, Good]
functionality_values_vec = [[-1,0,1]]
# Patient pain level: [MM, MS]
pain_level_values_vec = [[0.5, 0], [0.5, -0.25], [0.5, -0.5], [0.25, 0], [0.25, -0.25]]
# Action penalties
action_penalty(action_index, α = 1) = -α * (action_index == 1 ? 1 : 0)   # Penalize taking action "+2"
α_vec = [0.25, 0.5, 0.75, 1.0]
# Terminal rewards
term_reward_func_vec = [[-0.5, 0.75, 1], [-1, 0, 1]]   # [Poor, Acceptable, Good]
term_reward_pain_sev_vec = [[0.5, -0.5], [0.25, -0.25]]   # [MM, MS]

set_num = 1
mkpath("./Experiments/Policies/Set $set_num")

total_exp = length(functionality_values_vec) * length(pain_level_values_vec) * length(α_vec) * 
            length(term_reward_func_vec) * length(term_reward_pain_sev_vec)
subpath = "./Experiments/Policies/Set $set_num/"
exp_count = 0
for term_reward_func in term_reward_func_vec
    for term_reward_pain_sev in term_reward_pain_sev_vec
        for functionality_values in functionality_values_vec
            for pain_level_values in pain_level_values_vec
                for α in α_vec
                    global exp_count += 1
                    local R, u_terminal, u, π, opt_acts_mult

                    println("On experiment $exp_count of $total_exp")
                    R = contruct_R(states, actions, action_sets, α, state_health_partition, state_pain_partition)

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
                    _, _, opt_acts_mult = compute_policy_mult_actions(states, actions, action_sets, R, P, T; u_terminal = u_terminal)
                    length(opt_acts_mult)

                    ### Logging
                    filename_exp = subpath * "Experiment summary.txt"
                    fe = open(filename_exp, "a")

                    filename_policy = subpath * "Policy $exp_count.txt"
                    f = open(filename_policy, "w")

                    filename_policy_concat = subpath * "Experiment and policy summary.txt"
                    fc = open(filename_policy_concat, "a")

                    # Write to filename_exp
                    write(fe, "Experiment $exp_count\n")
                    write(fe, "u_terminal = $u_terminal\n")
                    write(fe, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
                    write(fe, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
                    write(fe, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
                    write(fe, "pain_level_values = $pain_level_values   # MM, MS\n")
                    write(fe, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
                    if length(opt_acts_mult) > 0
                        write(fe, "Multiple optimal actions at: $opt_acts_mult\n")
                    end
                    write(fe, "Total expected reward u = $(u[1])\n")
                    write(fe, "\n")
                    close(fe)

                    # Write to filename_policy
                    write(f, "Experiment $exp_count\n")
                    write(f, "u_terminal = $u_terminal\n")
                    write(f, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
                    write(f, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
                    write(f, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
                    write(f, "pain_level_values = $pain_level_values   # MM, MS\n")
                    write(f, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
                    if length(opt_acts_mult) > 0
                        write(fe, "Multiple optimal actions at: $opt_acts_mult\n")
                    end
                    write(f, "Total expected reward u = $(u[1])\n")
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
                    write(f, "\n")
                    close(f)
                    
                    # Write to filename_policy_concat
                    write(fc, "Experiment $exp_count\n")
                    write(fc, "u_terminal = $u_terminal\n")
                    write(fc, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
                    write(fc, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
                    write(fc, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
                    write(fc, "pain_level_values = $pain_level_values   # MM, MS\n")
                    write(fc, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
                    if length(opt_acts_mult) > 0
                        write(fc, "Multiple optimal actions at: $opt_acts_mult\n")
                    end
                    write(fc, "Total expected reward u = $(u[1])\n")
                    write(fc, "\n")
                    # Policy
                    write(fc, "\tT\t\t")
                    for t = 1:T
                        write(fc, "$t\t\t")
                    end
                    write(fc, "\n")
                    for i in eachindex(states)
                        write(fc, "$(states[i])\t")
                        for t in 1:T
                            write(fc, "$(actions[π[t][i]])\t\t")
                        end
                        write(fc, "\n")
                    end
                    write(fc, "\n")
                    close(fc)
                end
            end
        end
    end
end
