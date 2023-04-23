using XLSX
using DataFrames
using Pipe

data_xlsx = XLSX.readxlsx("patients-states.xlsx")
sheet = data_xlsx["Sheet1"]

states = ["[0,P,MM]", "[0,A,MM]", "[0,G,MM]", "[0,P,MS]", "[0,A,MS]", "[0,G,MS]",
    "[1,P,MM]", "[1,A,MM]", "[1,G,MM]", "[1,P,MS]", "[1,A,MS]", "[1,G,MS]",
    "[2,P,MM]", "[2,A,MM]", "[2,G,MM]", "[2,P,MS]", "[2,A,MS]", "[2,G,MS]" ]
num_states = length(states)
# state_map = Dict([i => state_desc[i] for i = 1:18])
# state_map[2]

# Create DataFames for boostrapped transition matrix, original transition matrix, and transition counts
col_names = ["$i" for i = 1:18]
df = @pipe DataFrame(sheet["V42:AM59"], col_names) |> convert.(Float64, _)
df_orig = @pipe DataFrame(sheet["V22:AM39"], col_names) |> convert.(Float64, _)
df_counts = @pipe DataFrame(sheet["V2:AM19"], col_names) |> convert.(Int, _)

##### Action space
actions = ["+2", "+1", "0" , "-1", "-2"]
num_actions = length(actions)

action_set = [(1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3),
              (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4),
              (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,4,5)]
action_set[1]
# For j ∈ action_set[i], state i can take action actions[j]

##### Transition probabilities
P_full = Matrix(df)
P_full

P = Vector{Matrix{Float64}}(undef, length(actions))
for i in eachindex(P)
    P[i] = zeros(size(P_full))
end

P[1][1:6, 13:18] = P_full[1:6, 13:18]
P[1]

P[2][1:12, 7:18] = P_full[1:12, 7:18]
P[2]

P[3] = copy(P_full)
P[3]

P[4][7:18, 1:12] = P_full[7:18, 1:12]
P[4]

P[5][13:18, 1:6] = P_full[13:18, 1:6]
P[5]

##### Rewards
# Patient functionality:
    # Poor = -1
    # Acceptable = 0
    # Good = 1
functionality_values = [-1,0,1]
# Patient pain level
    # MM = 0.5
    # MS = 0
pain_level_values = [0.5, 0]

reachable_states = map(a -> map(i ->  compute_reachable_states(P, i, a), eachindex(states)), eachindex(actions))
# filter!(vec -> !isempty(vec), reachable_states[1])
reachable_states[1]

states_with_poor = [i for i = 1:3:18]
states_with_acceptable = [i for i = 2:3:18]
states_with_good = [i for i = 3:3:18]
state_health_partition = [states_with_poor, states_with_acceptable, states_with_good]

states_with_MM = @pipe Iterators.flatten([[i; i+1; i+2] for i in 1:6:18]) |> collect(_)
states_with_MS = @pipe Iterators.flatten([[i; i+1; i+2] for i in 4:6:18]) |> collect(_)
state_pain_partition = [states_with_MM, states_with_MS]

R = zeros(num_actions, num_states, num_states)
for i in eachindex(states)
    for a_ind in action_set[i]
        # println("In state $i, taking action $(actions[a_ind]), able to reach states $(reachable_states[a_ind][i])\n")
        for j in reachable_states[a_ind][i]

            reward = 0

            ### Current state i
            health_list_ind_i = 0
            for k = eachindex(state_health_partition)
                if i in state_health_partition[k]
                    health_list_ind_i = k
                    break
                end
            end
            if health_list_ind_i == 0
                error("Current state $i not found in state_health_partition")
            end

            pain_list_ind_i = 0
            for k = eachindex(state_pain_partition)
                if i in state_pain_partition[k]
                    pain_list_ind_i = k
                    break
                end
            end
            if pain_list_ind_i == 0
                error("Current state $i not found in state_pain_partition")
            end

            ### Future state j
            health_list_ind_j = 0
            for k = eachindex(state_health_partition)
                if j in state_health_partition[k]
                    health_list_ind_j = k
                    break
                end
            end
            if health_list_ind_j == 0
                error("Future state $j not found in state_health_partition")
            end

            pain_list_ind_j = 0
            for k = eachindex(state_pain_partition)
                if j in state_pain_partition[k]
                    pain_list_ind_j = k
                    break
                end
            end
            if pain_list_ind_j == 0
                error("Future state $j not found in state_pain_partition")
            end

            reward += functionality_values[health_list_ind_j] - functionality_values[health_list_ind_i]
            reward += pain_level_values[pain_list_ind_j] - pain_level_values[pain_list_ind_i]

            R[a_ind,i,j] = reward
        end
    end
end
R
