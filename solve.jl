using LinearAlgebra
using Pipe

# Value Iteration

d = Vector{String}(undef, num_states)   # policy
v_current = zeros(num_states)   # arbitrary initial values
v_next = Vector{Float64}(undef, num_states)
max_iter = 5000
tol = 1e-5
n = 1

while n < max_iter
    for s in states
        reward_wait = R_pre[s] + λ * sum( P[s,j] * v_current[j] for j in states )
        reward_transplant = R_post[s]
        if reward_wait >= reward_transplant
            global v_next[s] = reward_wait
            global d[s] = "W"
        else
            global v_next[s] = reward_transplant
            global d[s] = "T"
        end
    end

    # if n % 100 == 0
    #     println("Norm: $(norm(v_next - v_current))")
    # end

    if norm(v_next - v_current) < tol * (1-λ)/(2λ)
        break
    end

    global v_current = copy(v_next)
    global n += 1
end

v1 = copy(v_current)
d1 = copy(d)