"""
Define rewards and compute respective policies.
"""

include("./parameters wl.jl")
include("./util.jl")
include("./solve.jl")

rows_dropped = "none"
method = "BR"
order = "ascending"

#### Rewards
work_dir = pwd()
folder = "Policy weight loss"
local_name_no_week_no_ext = "Rewards imputed $method order $order $rows_dropped dropped"
filename = joinpath(work_dir, folder, local_name_no_week_no_ext)

R = load_R(states, actions, action_sets, filename, T)

#### Compute policy
u, π = compute_policy(states, actions, action_sets, R, P, T)

permuted_indices = [3,2,1,6,5,4,9,8,7,12,11,10,15,14,13,18,17,16]
# new_states = states[permuted_indices]
π = map(t -> π[t][permuted_indices], 1:T)
u = map(t -> u[t][permuted_indices], 1:(T+1))

# Re-order states to be consistent with new presentation ordering
states = ["[0,MM,G]", "[0,MM,A]", "[0,MM,P]", "[0,MS,G]", "[0,MS,A]", "[0,MS,P]",
    "[1,MM,G]", "[1,MM,A]", "[1,MM,P]", "[1,MS,G]", "[1,MS,A]", "[1,MS,P]",
    "[2,MM,G]", "[2,MM,A]", "[2,MM,P]", "[2,MS,G]", "[2,MS,A]", "[2,MS,P]"]

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
push!(local_file_names, "Policy wl method $method order $order $rows_dropped dropped.txt")
for filename in local_file_names
    # if filename == "Policy $exp_count.txt"
    #     f = open(filename, "w")
    # else
    #     f = open(filename, "a")
    # end
    work_dir = pwd()
    folder = "Policy weight loss"
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

