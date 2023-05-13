"""
Simulate stochastic process.
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

##############################################################################

# set_num = 1
mkpath("./Experiments/Simulations/Set $set_num")

subpath = "./Experiments/Simulations/Set $set_num/"