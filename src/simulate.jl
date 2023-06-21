"""
Simulate stochastic process.
"""

include("./parameters.jl")
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

##############################################################################

# set_num = 1
# mkpath("./Experiments/Simulations/Set $set_num")

# subpath = "./Experiments/Simulations/Set $set_num/"