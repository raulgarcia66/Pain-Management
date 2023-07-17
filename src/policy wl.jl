"""
Define rewards and compute respective policies.
"""

include("./parameters.jl")
include("./util.jl")
include("./solve.jl")

rows_dropped = "none"
method = "LR"
order = "ascending"

#### Rewards
filename_no_week_no_ext = "Rewards imputed $method order $order $rows_dropped dropped"
R = load_R(states, actions, action_sets, filename_no_week_no_ext, T)

#### Compute policy
u, π = compute_policy(states, actions, action_sets, R, P, T)

# t = 1
# i = 5
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

