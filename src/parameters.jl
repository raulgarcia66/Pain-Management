"""
Compute parameters for pain management MDP framework.
"""
using XLSX
using DataFrames
using Pipe
include("./util.jl")

# Load data
data_xlsx = XLSX.readxlsx("patients-states.xlsx")
sheet = data_xlsx["Sheet1"]

# Create DataFames for boostrapped transition matrix, original transition matrix, and transition counts
col_names = ["$i" for i = 1:18]
df = @pipe DataFrame(sheet["V42:AM59"], col_names) |> convert.(Float64, _)
df_orig = @pipe DataFrame(sheet["V22:AM39"], col_names) |> convert.(Float64, _)
df_counts = @pipe DataFrame(sheet["V2:AM19"], col_names) |> convert.(Int, _)

##### Horizon
T = 6   # num decision epochs (⟹ horizon is T+1)


##### States
states = ["[0,P,MM]", "[0,A,MM]", "[0,G,MM]", "[0,P,MS]", "[0,A,MS]", "[0,G,MS]",
    "[1,P,MM]", "[1,A,MM]", "[1,G,MM]", "[1,P,MS]", "[1,A,MS]", "[1,G,MS]",
    "[2,P,MM]", "[2,A,MM]", "[2,G,MM]", "[2,P,MS]", "[2,A,MS]", "[2,G,MS]"]
num_states = length(states)
# state_map = Dict([i => state_desc[i] for i = 1:18])


##### Action space
actions = ["+2", "+1", "0" , "-1", "-2"]
num_actions = length(actions)

action_sets = [(1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3),
              (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3,4),
              (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,4,5), (3,5)]  # state 18 modified
action_sets[1]
# action_sets[i] is the indices of the actions that state i can take


##### Transition probabilities
using_orig_matrix = false
if !using_orig_matrix
    P_full = Matrix(df)
else
    P_full = Matrix(df_orig)
end
# TODO: P_full[13:18,7:12] is all zeros, same for P_full_orig and trans_counts
P_full_orig = Matrix(df_orig)
trans_counts = Matrix(df_counts)

# Create matrix for each action by grabbing the portion that corresponds to the action
# and zero-ing out all other transitions (non-reachable)
P = Vector{Matrix{Float64}}(undef, length(actions))
for i in eachindex(P)
    P[i] = zeros(size(P_full))
end

P[1][1:6, 13:18] = P_full[1:6, 13:18]
P[1]
# for row in eachrow(P[1]) println("$(sum(row))") end

P[2][1:6, 7:12] = P_full[1:6, 7:12]
P[2][7:12, 13:18] = P_full[7:12, 13:18]
P[2]
# for row in eachrow(P[2]) println("$(sum(row))") end

P[3][1:6, 1:6] = P_full[1:6, 1:6]
P[3][7:12, 7:12] = P_full[7:12, 7:12]
P[3][13:18, 13:18] = P_full[13:18, 13:18]
P[3]
# for row in eachrow(P[3]) println("$(sum(row))") end

P[4][7:12, 1:6] = P_full[7:12, 1:6]
P[4][13:18, 7:12] = P_full[13:18, 7:12]
P[4]
# for row in eachrow(P[4]) println("$(sum(row))") end

P[5][13:18, 1:6] = P_full[13:18, 1:6]
P[5]
# for row in eachrow(P[5]) println("$(sum(row))") end

# for i in eachindex(P)
#     println("\nMatrix for action $(actions[i])")
#     for row in eachrow(P[i])
#         println("$(sum(row))")
#     end
# end


##### Rewards
# See policy.jl
println("Executed parameters.jl")