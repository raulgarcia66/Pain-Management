"""
Define rewards and compute respective policies.
"""

include("./parameters bmi.jl")
include("./util.jl")
include("./solve.jl")

method = "LR"

#### Rewards
work_dir = pwd()
folder = "Policy bmi"
# local_name = "rewards_all_weeks_BMI.xlsx"
local_name = "Rewards BMI imputed method $method.xlsx"
filename = joinpath(work_dir, folder, local_name)
R = load_R_bmi(states, actions, action_sets, filename, T)

#### Compute policy
u, π = compute_policy(states, actions, action_sets, R, P, T)

# t = 1
# i = 5
# u[t][i]
# π[t][i]
# actions[π[t][i]]

policy_matrix = hcat(map(t -> actions[π[t]], 1:T)...)

#### Logging
local_file_names = String[]
# push!(local_file_names, "Experiment summary.txt")
# push!(local_file_names, "Experiment and policy summary.txt")
push!(local_file_names, "Policy bmi method $method.txt")
for filename in local_file_names
    # if filename == "Policy $exp_count.txt"
    #     f = open(filename, "w")
    # else
    #     f = open(filename, "a")
    # end
    work_dir = pwd()
    folder = "Policy bmi"
    fullpath = joinpath(work_dir, folder, filename)
    f = open(fullpath, "w")

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
