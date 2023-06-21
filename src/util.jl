"""
Helper/utility functions.
"""

using XLSX
using DataFrames
using Pipe

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
    # R[a,i,j] is dimension num_actions × num_states × num_states
    num_states = length(states)
    num_actions = length(actions)
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
    # R[a,i,j] is dimension num_actions × num_states × num_states
    num_states = length(states)
    num_actions = length(actions)
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

"""
DESCRIPTION.
"""
function write_R(R::Array{W,3}, states, actions, filename) where W
    ### Example 1
    # XLSX.openxlsx("my_new_file.xlsx", mode="w") do xf
    #     sheet = xf[1]
    #     XLSX.rename!(sheet, "new_sheet")
    #     sheet["A1"] = "this"
    #     sheet["A2"] = "is a"
    #     sheet["A3"] = "new file"
    #     sheet["A4"] = 100
    
    #     # will add a row from "A5" to "E5"
    #     sheet["A5"] = collect(1:5) # equivalent to `sheet["A5", dim=2] = collect(1:4)`
    
    #     # will add a column from "B1" to "B4"
    #     sheet["B1", dim=1] = collect(1:4)
    
    #     # will add a matrix from "A7" to "C9"
    #     sheet["A7:C9"] = [ 1 2 3 ; 4 5 6 ; 7 8 9 ]
    # end

    ### Example 2
    # filename = "myfile.xlsx"

    # # Some example data to try writing to .xlsx
    # columns = Vector()
    # push!(columns, [1, 2, 3])
    # push!(columns, ["a", "b", "c"])
    # labels = [ "column_1", "column_2"]

    # XLSX.openxlsx(filename, mode="w") do xf
    #     sheet = xf[1]
    #     # Write our data to sheet 1
    #     XLSX.writetable!(sheet, columns, labels, anchor_cell=XLSX.CellRef("A1"))
    #     # Write the same data, but to a different place in the sheet
    #     XLSX.writetable!(sheet, columns, labels, anchor_cell=XLSX.CellRef("D1"))
    #     # Add a new sheet, which we will then access with xf[2]
    #     XLSX.addsheet!(xf)
    #     # Write the same data, but to sheet 2, in yet another position
    #     XLSX.writetable!(xf[2], columns, labels, anchor_cell=XLSX.CellRef("B2"))
    # end

    #########################################################################
    # R[a,i,j] is dimension num_actions × num_states × num_states
    num_actions = length(actions)

    rows = [i for i = 2:22:(22*num_actions)]
    XLSX.openxlsx(filename, mode="w") do xf
        sheet = xf[1]
        for a in eachindex(actions)
            sheet["B" * string(rows[a])] = "Action: " * actions[a]
            sheet["C" * string(rows[a]+1)] = states   # add states to row; equivalent to `sheet[cell index, dim=2]
            sheet["B" * string(rows[a]+2), dim=1] = states   # add states to column
            sheet["C" * string(rows[a]+2) * ":" * "T" * string(rows[a]+1+18)] = R[a,:,:]
        end
    end
end

"""
DESCRIPTION.
"""
function write_R(R::Vector{Array{W,2}}, states, actions, filename) where W
    # R[t][i,a]
    num_actions = length(actions)

    rows = [i for i = 2:22:(22*num_actions)]
    # XLSX.openxlsx(filename, mode="w") do xf
    #     sheet = xf[1]
    #     for a in eachindex(actions)
    #         sheet["B" * string(rows[a])] = "Action: " * actions[a]
    #         sheet["C" * string(rows[a]+1)] = states   # add states to row; equivalent to `sheet[cell index, dim=2]
    #         sheet["B" * string(rows[a]+2), dim=1] = states   # add states to column
    #         sheet["C" * string(rows[a]+2) * ":" * "T" * string(rows[a]+1+18)] = R[a,:,:]
    #     end
    # end
end

"""
DESCRIPTION.
"""
function load_R(states, actions, action_sets, filename_no_week_no_ext, T::Int)
    num_states = length(states)
    num_actions = length(actions)
    R = [zeros(num_states, num_actions) for _ in 1:T]

    # Read in for one time period
    for t = 1:T
        data_xlsx = XLSX.readxlsx(filename_no_week_no_ext * " week $t.xlsx")
        sheet = data_xlsx["Rewards"]

        col_names = ["$i" for i = 1:3]
        df = @pipe DataFrame(sheet["A2:C19"], col_names) |> convert.(Float64, _)
        mat =  Matrix(df)
        mat = mat[:, end:-1:1]

        for i = 1:num_states
            for a in eachindex(action_sets[i])
                R[t][i,action_sets[i][a]] = copy(mat[i,a])
            end
        end
    end

    return R
end
