"""
Compute parameters for pain management MDP framework.
"""

# States use BMI and weight loss
using XLSX
using DataFrames
using Pipe
include("./util.jl")

#### Load data
work_dir = pwd()
folder = "Data for states and transitions bmi mdasi weight loss"
data_xlsx = XLSX.readxlsx(joinpath(work_dir, folder, "Parameters - Reward model with mdasi and weight loss.xlsx"))
sheet = data_xlsx["transition probabilities"]

# Create DataFames for boostrapped transition matrix, original transition matrix, and transition counts
P_full = @pipe DataFrame(sheet["B3:S20"], string.(collect(1:18))) |> mapcols(col -> replace(col, "NaN" => NaN), _) |> Matrix{Float64}(_)

##### Horizon
T = 6   # num decision epochs (⟹ horizon is T+1)


##### States
# states = ["[0,P,MM]", "[0,A,MM]", "[0,G,MM]", "[0,P,MS]", "[0,A,MS]", "[0,G,MS]",
#     "[1,P,MM]", "[1,A,MM]", "[1,G,MM]", "[1,P,MS]", "[1,A,MS]", "[1,G,MS]",
#     "[2,P,MM]", "[2,A,MM]", "[2,G,MM]", "[2,P,MS]", "[2,A,MS]", "[2,G,MS]"]
states = ["[0,MM,G]", "[0,MM,A]", "[0,MM,P]", "[0,MS,G]", "[0,MS,A]", "[0,MS,P]",
    "[1,MM,G]", "[1,MM,A]", "[1,MM,P]", "[1,MS,G]", "[1,MS,A]", "[1,MS,P]",
    "[2,MM,G]", "[2,MM,A]", "[2,MM,P]", "[2,MS,G]", "[2,MS,A]", "[2,MS,P]"]
num_states = length(states)
# state_map = Dict([i => state_desc[i] for i = 1:18])


##### Action space
actions = ["+2", "+1", "0" , "-1", "-2"]
num_actions = length(actions)

# actions also omitted if computed trans. prob. are 0
# From NaNs in transition probabilities
    # states 4,5 do not have "+2"
    # states 11,14,16 do not have "-1"
    # states 14 does not have "-2"
action_sets = [(1,2,3), (1,2,3), (1,2,3), (2,3), (2,3), (1,2,3),
              (2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3), (2,3,4),
              (3,4,5), (3,), (3,4,5), (3,5), (3,4,5), (3,5)]
action_sets[1]
# action_sets[i] is the indices of the actions that state i can take

# Then, remove actions due to NaNs in rewards
action_sets_by_week = Vector{Vector{Union{Tuple{Int, Vararg{Int}}, Tuple{}}}}(undef, T)
action_sets_by_week[1] = [(2,3), (1,2,3), (1,2,3), (3,), (2,), (3,),
                        (3,), (3,4), (2,3,4), (3,), (), (3,4),
                        (3,), (3,), (3,4), (), (3,), (3,)]
action_sets_by_week[2] = [(1,2,3), (), (1,2,3), (), (), (1,2,3),
                        (3,), (2,3), (2,3,4), (), (), (2,3),
                        (3,4), (), (3,4), (), (), (3,5)]
action_sets_by_week[3] = [(), (), (), (2,3), (), (1,2,3),
                        (), (), (), (2,3), (), (2,3,4),
                        (), (), (), (3,), (3,), (3,5)]
action_sets_by_week[4] = [(3,), (3,), (1,2,3), (), (3,), (2,3),
                        (3,4), (2,3), (2,3,4), (3,), (2,3), (2,3,4),
                        (3,), (), (3,4,5), (3,), (3,), (3,)]
action_sets_by_week[5] = [(1,2,3), (), (1,2,3), (), (), (1,2,3),
                        (2,3), (2,3,4), (2,3,4), (2,), (3,), (2,3,4),
                        (3,5), (3,), (3,4,5), (3,), (3,4), (3,5)]
action_sets_by_week[6] = [(3,), (3,), (1,2,3), (), (), (1,2,3),
                        (3,4), (2,3,4), (2,3,4), (4,), (), (2,3,4),
                        (5,), (), (3,4,5), (3,5), (5,), (3,5)]

##### Transition probabilities
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
println("Parameters set.")