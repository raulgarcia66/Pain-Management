"""
Define rewards and compute respective policies.
"""

include("./parameters bmi.jl")
include("./util.jl")
include("./solve.jl")

#### Rewards
work_dir = pwd()
folder = "Data for states and transitions bmi"
local_name = "rewards_all_weeks_BMI.xlsx"
filename = joinpath(work_dir, folder, filename)
R = load_R_bmi(states, actions, action_sets, filename, T)

#### Compute policy
u, π = compute_policy(states, actions, action_sets, R, P, T)

# t = 1
# i = 5
# u[t][i]
# π[t][i]
# actions[π[t][i]]

policy_matrix = hcat(map(t -> actions[π[t]], 1:T)...)