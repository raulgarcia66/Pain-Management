"""
Define rewards and compute respective policies.
"""

# Rewards use MDASI scores and weight loss
include("./parameters bmi mdasi wl.jl")
# include("./util.jl")
include("./solve.jl")

#### Rewards
work_dir = pwd()
folder = "Data for states and transitions bmi mdasi weight loss"
local_name = "Parameters - Reward model with mdasi and weight loss.xlsx"
filename = joinpath(work_dir, folder, local_name)
R = load_R_mdasi_wl(states, actions, action_sets, filename, T)