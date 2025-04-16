import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parameters
del_x = 0.1
n = 100
r_w = 0.10
r_out = 100.0
Peclet = 10
gamma = 0.1

T_low = 0
T_high = 1
del_t = 0.00001
ITER = 500000

# grid and initial Condition
r_array = np.linspace(r_w, r_out, n)
T = T_high * np.ones(n)
T[0] = T_low

# coefficients
a_array = del_t / (Peclet * r_array * del_x)
b_array = del_t / (Peclet * del_x**2) * np.ones(n)
c_array = -gamma * del_t / (r_array * del_x)

n_interior = n - 2

main_diag = 1 + a_array[1:-1] + 2 * b_array[1:-1] + c_array[1:-1]
lower_diag = -b_array[2:-1]                         # length = n_interior - 1
upper_diag = -a_array[1:-2] - b_array[1:-2] - c_array[1:-2]  # length = n_interior - 1

A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')

history = [T.copy()]
for it in range(ITER):
    T[0] = T_low
    T[-1] = T[-2]

    rhs = T[1:-1].copy()
    rhs[0] += b_array[1] * T[0]
    rhs[-1] += (a_array[-2] + b_array[-2] + c_array[-2]) * T[-1]

    T_interior_new = spsolve(A, rhs)
    T[1:-1] = T_interior_new

    if it % (ITER // 100) == 0:
        history.append(T.copy())

history = np.array(history)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(r_array, T, label='Final Temperature (Backward Euler)')
plt.xlabel('Radial Distance r')
plt.ylabel('Temperature')
plt.title('Final Temperature Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Analytical comparison
T_steady = 1 - (r_w / r_array) ** (gamma * Peclet)
plt.plot(r_array, T, label="Numerical Final T (Backward Euler)")
plt.plot(r_array, T_steady, '--', label="Analytical Steady-State", alpha=0.8)
plt.legend()
plt.xlabel("r")
plt.ylabel("T")
plt.title("Numerical vs. Analytical Steady-State")
plt.grid(True)
plt.show()

# Heatmaps
time_total = ITER * del_t
plt.figure(figsize=(12,6))
plt.imshow(history, extent=[r_w, r_out, 0, time_total], origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(label='Temperature')
plt.xlabel('Radial Distance r')
plt.ylabel('Time')
plt.title('Heatmap of Temperature Distribution (Backward Euler)')
plt.show()

# plt.figure(figsize=(12,6))
# plt.imshow(history, extent=[r_w, r_out, 0, time_total], origin='lower', aspect='auto', cmap='hot')
# plt.colorbar(label='Temperature')
# plt.xlabel('Radial Distance r')
# plt.ylabel('Time')
# plt.title('Heatmap of Temperature Distribution (Backward Euler)')
# plt.show()

# Video animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(r_w, r_out)
ax.set_ylim(T_low, T_high + 0.1)
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

ani.save("temperature_evolution_backward_euler.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
