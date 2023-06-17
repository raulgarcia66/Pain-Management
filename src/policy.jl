"""
Define rewards and compute respective policies.
"""

include("./parameters.jl")

# states_with_poor = [i for i = 1:3:18]
# states_with_acceptable = [i for i = 2:3:18]
# states_with_good = [i for i = 3:3:18]
# state_health_partition = [states_with_poor, states_with_acceptable, states_with_good]

# states_with_MM = @pipe Iterators.flatten([[i; i+1; i+2] for i in 1:6:18]) |> collect(_)
# states_with_MS = @pipe Iterators.flatten([[i; i+1; i+2] for i in 4:6:18]) |> collect(_)
# state_pain_partition = [states_with_MM, states_with_MS]

# ##### Rewards
# # Patient functionality: [Poor, Acceptable, Good]
# functionality_values = [-2.0, 0.25, 1.0]

# # Patient pain level: [MM, MS]
# pain_level_values = [0.25, -0.1]

# Action penalties
# action_penalty(action_index, α = 1) = -α * (action_index == 1 ? 1 : 0)   # Penalize taking action "+2"
# α = 0.0

# Reward matrix
# rewards_transition = false   # if true, use rewards that depend on transitions
# if rewards_transition
#     R = construct_R_transition(states, actions, action_sets, P, α, state_health_partition, state_pain_partition,
#                     functionality_values, pain_level_values, action_penalty)
# else
#     R = construct_R_state_occupation(states, actions, action_sets, P, α, state_health_partition, state_pain_partition, 
#                     functionality_values, pain_level_values, action_penalty)
# end

# write_R(R, states, actions, "Rewards transition.xlsx")
# write_R(R, states, actions, "Rewards state occupation.xlsx")

##### Compute policy
# term_reward_func = [0.0, 0.0, 0.0]   # [Poor, Acceptable, Good]
# term_reward_pain_sev = [0.0, 0.0]   # [MM, MS]
# u_terminal = map(i -> begin
#     local u = 0
#     if i in states_with_good
#         u += term_reward_func[3]
#     elseif i in states_with_acceptable
#         u += term_reward_func[2]
#     elseif i in states_with_poor
#         u += term_reward_func[1]
#     end
#     if i in states_with_MM
#         u += term_reward_pain_sev[1]
#     elseif i in states_with_MS
#         u += term_reward_pain_sev[2]
#     end
#     return u
# end, eachindex(states))

# u, π = compute_policy(states, actions, action_sets, R, P, T; u_terminal = u_terminal)
# u_mult, π_mult, opt_acts_mult = compute_policy_mult_actions(states, actions, action_sets, R, P, T; u_terminal = u_terminal)
# d
# u[1]
# π[1][1]
# π_mult[1][1]

# local_file_names = String[]
# push!(local_file_names, "Experiment summary.txt")
# push!(local_file_names, "Experiment and policy summary.txt")
# push!(local_file_names, "Policy $exp_count.txt")
# for filename in local_file_names
#     if filename == "Policy $exp_count.txt"
#         f = open(filename, "w")
#     else
#         f = open(filename, "a")
#     end

#     write(f, "Experiment $exp_count\n")
#     if rewards_transition
#         write(f, "Rewards coupled with transitioning.\n")
#     else
#         write(f, "Rewards for state occupation each period.\n")
#     end
#     if using_orig_matrix
#         write(f, "Using original matrix.\n")
#     end
#     write(f, "u_terminal = $u_terminal\n")
#     write(f, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
#     write(f, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
#     write(f, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
#     write(f, "pain_level_values = $pain_level_values   # MM, MS\n")
#     write(f, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
#     if length(opt_acts_mult) > 0
#         write(f, "Multiple optimal actions at: $opt_acts_mult\n")
#     end
#     write(f, "Total expected reward u = $(u[1])\n")
#     write(f, "\n")

#     if filename == "Experiment summary.txt"
#         close(f)
#     else
#         # Policy
#         write(f, "\tT\t\t")
#         for t = 1:T
#             write(f, "$t\t\t")
#         end
#         write(f, "\n")
#         for i in eachindex(states)
#             write(f, "$(states[i])\t")
#             for t in 1:T
#                 write(f, "$(actions[π[t][i]])\t\t")
#             end
#             write(f, "\n")
#         end
#         write(f, "\n")
#         close(f)
#     end
# end

##############################################################################
rows_dropped = "none"
method = "LR"
order = "ascending"

#### Rewards
filename_no_week_no_ext = "Rewards imputed $method order $order $rows_dropped dropped"
R = load_R(states, actions, action_sets, filename_no_week_no_ext, T)

#### Compute policy
u, π = compute_policy_time_dep_rewards(states, actions, action_sets, R, P, T)

# t = 1; i = 5;
# u[t][i]
# π[t][i]
# actions[π[t][i]]

# policy_matrix = hcat(map(t -> actions[π[t]], 1:T)...)

#### Logging

local_file_names = String[]
# push!(local_file_names, "Experiment summary.txt")
# push!(local_file_names, "Experiment and policy summary.txt")
push!(local_file_names, "Policy $method order $order $rows_dropped dropped.txt")
for filename in local_file_names
    # if filename == "Policy $exp_count.txt"
    #     f = open(filename, "w")
    # else
    #     f = open(filename, "a")
    # end
    f = open(filename, "w")

    write(f, "Method: $method\nOrder: $order\nRows dropped: $rows_dropped\n")
    # if using_orig_matrix
    #     write(f, "Using original matrix.\n")
    # end
    # write(f, "u_terminal = $u_terminal\n")
    # if length(opt_acts_mult) > 0
    #     write(f, "Multiple optimal actions at: $opt_acts_mult\n")
    # end
    write(f, "Total expected reward u = $(u[1])\n")
    write(f, "\n")

    if filename == "Experiment summary.txt"
        close(f)
    else
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
    end
end

##############################################################################
# ##### Experiments in loop

# # Patient functionality: [Poor, Acceptable, Good]
# functionality_values_vec = [[-2, 0.5, 1], [-2, 0.25, 1], [-2, 0, 1]]
# # Patient pain level: [MM, MS]
# pain_level_values_vec = [[0.25, 0], [0.25, -0.10], [0.15, 0], [0.15, -0.10]]
# # Action penalties
# action_penalty(action_index, α = 1) = -α * (action_index == 1 ? 1 : 0)   # Penalize taking action "+2"
# α_vec = [0.0, 0.05, 0.10, 0.15, 0.20]
# # Terminal rewards
# term_reward_func_vec = [[0.0, 0.0, 0.0]]   # [Poor, Acceptable, Good]
# term_reward_pain_sev_vec = [[0.0, 0.0]]   # [MM, MS]
# # Reward matrix
# rewards_transition = false   # if true, use rewards that depend on transitions

# set_num = 12
# mkpath("./Experiments/Policies/Set $set_num")
# subpath = "./Experiments/Policies/Set $set_num/"

# # Parameter summary
# local_file_names = String[]
# push!(local_file_names, "Experiment summary.txt")
# push!(local_file_names, "Experiment and policy summary.txt")
# push!(local_file_names, "Sets summary.txt")
# for filename in local_file_names
#     if filename == "Sets summary.txt"
#         file = "./Experiments/Policies/" * filename
#         f = open(file, "a")
#     else
#         file = subpath * filename
#         f = open(file, "w")
#     end
#     write(f, "Experiment set $set_num\n")
#     if rewards_transition
#         write(f, "Rewards coupled with transitioning.\n")
#     else
#         write(f, "Rewards for state occupation each period.\n")
#     end
#     if using_orig_matrix
#         write(f, "Using original matrix.\n")
#     end
#     write(f, "Parameter sets:\n")
#     write(f, "functionality_values_vec = $functionality_values_vec\n")
#     write(f, "pain_level_values_vec = $pain_level_values_vec\n")
#     write(f, "α_vec = $α_vec\n")
#     write(f, "term_reward_func_vec = $term_reward_func_vec\n")
#     write(f, "term_reward_pain_sev_vec = $term_reward_pain_sev_vec\n\n\n")
#     close(f)
# end

# # Compute policies
# total_exp = length(functionality_values_vec) * length(pain_level_values_vec) * length(α_vec) * 
#             length(term_reward_func_vec) * length(term_reward_pain_sev_vec)
# exp_count = 0
# for term_reward_func in term_reward_func_vec
#     for term_reward_pain_sev in term_reward_pain_sev_vec
#         for functionality_values in functionality_values_vec
#             for pain_level_values in pain_level_values_vec
#                 for α in α_vec
#                     global exp_count += 1
#                     local R, u_terminal, u, π, opt_acts_mult

#                     println("On experiment $exp_count of $total_exp")
#                     if rewards_transition
#                         R = construct_R_transition(states, actions, action_sets, P, α, state_health_partition, state_pain_partition,
#                                         functionality_values, pain_level_values, action_penalty)
#                     else
#                         R = construct_R_state_occupation(states, actions, action_sets, P, α, state_health_partition, state_pain_partition, 
#                                         functionality_values, pain_level_values, action_penalty)
#                     end

#                     u_terminal = map(i -> begin
#                         local u = 0
#                         if i in states_with_good
#                             u += term_reward_func[3]
#                         elseif i in states_with_acceptable
#                             u += term_reward_func[2]
#                         elseif i in states_with_poor
#                             u += term_reward_func[1]
#                         end
#                         if i in states_with_MM
#                             u += term_reward_pain_sev[1]
#                         elseif i in states_with_MS
#                             u += term_reward_pain_sev[2]
#                         end
#                         return u
#                     end, eachindex(states))

#                     u, π = compute_policy(states, actions, action_sets, R, P, T; u_terminal = u_terminal)
#                     _, _, opt_acts_mult = compute_policy_mult_actions(states, actions, action_sets, R, P, T; u_terminal = u_terminal)

#                     ### Logging                    
#                     local_file_names = String[]
#                     push!(local_file_names, "Experiment summary.txt")
#                     push!(local_file_names, "Experiment and policy summary.txt")
#                     push!(local_file_names, "Policy $exp_count.txt")

#                     for filename in local_file_names
#                         file = subpath * filename
#                         if file == subpath * "Policy $exp_count.txt"
#                             f = open(file, "w")
#                         else
#                             f = open(file, "a")
#                         end

#                         write(f, "Experiment $exp_count\n")
#                         if rewards_transition
#                             write(f, "Rewards coupled with transitioning.\n")
#                         else
#                             write(f, "Rewards for state occupation each period.\n")
#                         end
#                         if using_orig_matrix
#                             write(f, "Using original matrix.\n")
#                         end
#                         write(f, "u_terminal = $u_terminal\n")
#                         write(f, "term_reward_func = $term_reward_func   # Poor, Acceptable, Good\n")
#                         write(f, "term_reward_pain_sev = $term_reward_pain_sev   # MM, MS\n")
#                         write(f, "functionality_values = $functionality_values   # Poor, Acceptable, Good\n")
#                         write(f, "pain_level_values = $pain_level_values   # MM, MS\n")
#                         write(f, "α = $α   # Penalty of -α if action is + 2, 0 o.w.\n")
#                         if length(opt_acts_mult) > 0
#                             write(f, "Multiple optimal actions at: $opt_acts_mult\n")
#                         end
#                         write(f, "Total expected reward u = $(u[1])\n")
#                         write(f, "\n")

#                         if file == subpath * "Experiment summary.txt"
#                             close(f)
#                         else
#                             # Policy
#                             write(f, "\tT\t\t")
#                             for t = 1:T
#                                 write(f, "$t\t\t")
#                             end
#                             write(f, "\n")
#                             for i in eachindex(states)
#                                 write(f, "$(states[i])\t")
#                                 for t in 1:T
#                                     write(f, "$(actions[π[t][i]])\t\t")
#                                 end
#                                 write(f, "\n")
#                             end
#                             write(f, "\n")
#                             close(f)
#                         end
#                     end

#                 end
#             end
#         end
#     end
# end

