import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import os

# non-dimensional parameters
del_x = 0.1            # spatial step
r_w = 1              # radius lower part of cilinder
r_out = 6           # some distance away from the cilinder
n = int((r_out - r_w) / del_x)                # total grid points
Peclet = 4.2
gamma = 0.1
T_low = 0
T_high = 1
del_t = 0.0001        # time step
ITER = 2000000        # number of time steps
final_time= ITER * del_t
r_array = np.linspace(r_w, r_out, n)

# dimensional parameter (p for physics)
Tp_high = 80
Tp_low = 20
rp_w = 0.05
rp_out = rp_w * r_out
rp_array = rp_w * r_array
up_inj = 0.05
final_time_p = rp_w / up_inj * ITER * del_t
print(f"Final time: {final_time:.2f} seconds")

# grid and initial Condition
T = T_high * np.ones(n)
T[0] = T_low

# coefficient arrays for the Forward Euler
a_array = del_t / (Peclet * r_array * del_x)         # appears in the T[i+1] term
b_array = del_t / (Peclet * del_x**2) * np.ones(n)      # appears in the T[i-1] term
c_array = -gamma * del_t / (r_array * del_x)            # additional term for T[i+1]

# we update only the interior nodes (indices 1 to n-2), vector will only contain unknown nodes.
n_interior = n - 2

main_diag = 1 - a_array[1:-1] - 2 * b_array[1:-1] - c_array[1:-1]
lower_diag = b_array[2:-1]                    # length = n_interior - 1
upper_diag = a_array[1:-2] + b_array[1:-2] + c_array[1:-2]  # length = n_interior - 1

A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')

history = [(Tp_high - Tp_low) * T.copy() +Tp_low]

for it in range(ITER):
    T[0] = T_low         # enforce inner Dirichlet BC
    T[-1] = T[-2]        # enforce outer Neumann BC
    T_interior = T[1:-1].copy()
    T_interior_new = A.dot(T_interior)

    # add boundary contributions:
    T_interior_new[0] += b_array[1] * T[0]
    T_interior_new[-1] += (a_array[-2] + b_array[-2] + c_array[-2]) * T[-1]

    T[1:-1] = T_interior_new
    
    # only record every now and then
    if it % (ITER // 100) == 0:
        T_converted = (Tp_high - Tp_low) * T +Tp_low

        # history.append(T.copy())

        history.append(T_converted.copy())

history = np.array(history)

print(f"Actual Final time: {final_time_p:.2f} seconds")
r_array = rp_w * r_array

# plotting
plt.figure(figsize=(10,6))
plt.plot(r_array, history[-1], label='Final Temperature')
plt.xlabel('Radial Distance r')
plt.ylabel('Temperature')
plt.title('Temperature Distribution (Final State)')
plt.legend()
plt.grid(True)
plt.show()

T_steady = 1 - (r_w / r_array) ** (gamma * Peclet)

plt.plot(r_array, history[-1], label="Numerical Final T")
plt.plot(r_array, T_steady, '--', label="Analytical Steady-State", alpha=0.8)
plt.legend()
plt.xlabel("r")
plt.ylabel("T")
plt.title("Numerical vs. Analytical Steady-State")
plt.grid(True)
plt.show()



time_total = ITER * del_t
# plt.figure(figsize=(12,6))
# plt.imshow(history, extent=[r_w, r_out, 0, time_total], origin='lower', aspect='auto', cmap='plasma')
# plt.colorbar(label='Temperature')
# plt.xlabel('Radial Distance r')
# plt.ylabel('Time')
# plt.title('Heatmap of Temperature Distribution')
# plt.show()

plt.figure(figsize=(12,6))
plt.imshow(history, extent=[r_w, r_out, 0, time_total], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('Radial Distance r')
plt.ylabel('Time')
plt.title('Heatmap of Temperature Distribution')
plt.show()



####################### video code
# Set up figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(rp_w, rp_out)
ax.set_ylim(Tp_low, Tp_high + 0.1)
ax.set_xlabel('Radial Distance r')
ax.set_ylabel('Temperature')
ax.set_title('Temperature Evolution Over Time')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(r_array, history[i])
    ax.set_title(f'Temperature Evolution - Time Step {i+1}/{len(history)}')
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(history), init_func=init,
                              blit=True, repeat=False)

# Save the animation as MP4
output_path = "temperature_evolution.mp4"
ani.save(output_path, fps=10, extra_args=['-vcodec', 'libx264'])