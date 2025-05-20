import numpy as np
from scipy import special
from scipy.sparse import diags
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dimensional parameters
r_w = 0.10
r_out = 10.0
u_inj = 3.17e-5
alpha = 6.8e-7
phi = 0.2
c_w = 4.18e3
rho_w = 1000
c_s = 0.8e3
rho_s = 2600

# Non-dimensional Parameters
n = 200
r_star_max = r_out / r_w
r_array = np.linspace(1.0, r_star_max, n)
del_r_star = r_array[1] - r_array[0]
del_t_star = 0.005
Peclet = (u_inj * r_w) / alpha
epsilon = 1.0 / Peclet
gamma = c_w * rho_w / (c_w * rho_w * phi + c_s * rho_s * (1 - phi))
T_low = 0.0
T_high = 1.0

print(f"del_r_star = {del_r_star:.2e}")
print(f"Peclet = {Peclet:.2e}, ε = {epsilon:.2e}, γ = {gamma:.2f}")

# Set Initial Condition
T = T_high * np.ones(n)
T[0] = T_low

ITER = 50000
a_array = del_t_star / (Peclet * r_array * del_r_star)
b_array = del_t_star / (Peclet * del_r_star**2) * np.ones(n)
c_array = -gamma * del_t_star / (r_array * del_r_star)

main_diag = 1 - a_array[1:-1] - 2 * b_array[1:-1] - c_array[1:-1]
lower_diag = b_array[2:-1]
upper_diag = a_array[1:-2] + b_array[1:-2] + c_array[1:-2]

A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')
history = [T.copy()]
for it in range(ITER):
    T[0] = T_low         # Dirichlet BC
    T[-1] = T[-2]        # Neumann BC
    T_interior = T[1:-1].copy()
    T_interior_new = A.dot(T_interior)

    # boundary terms
    T_interior_new[0] += b_array[1] * T[0]
    T_interior_new[-1] += (a_array[-2] + b_array[-2] + c_array[-2]) * T[-1]

    T[1:-1] = T_interior_new
    if it % (ITER // 500) == 0:
        history.append(T.copy())

history = np.array(history)

# Analytical erf-front approximate solution
t_total = ITER * del_t_star
times = np.linspace(0, t_total, len(history))

history_ana = []
r_fronts = []
for t_star in times:
    r_front = np.sqrt(1 + 2 * gamma * t_star)
    r_fronts.append(r_front)
    a = 1.0 / np.sqrt(2 * epsilon * t_star + 1e-12)
    # a = 1.0 / np.sqrt(2 * epsilon * t_star*(1-0.5*gamma*t_star) + 1e-12)
    z = a * (r_array - r_front)
    T_approx = 0.5 * (1 + special.erf(z))
    history_ana.append(T_approx)

history_ana = np.array(history_ana)
r_fronts = np.array(r_fronts)
history_step = [(r_array > r_f).astype(float) for r_f in r_fronts]

# Plot final state
plt.figure(figsize=(10, 6))
plt.plot(r_array, history[-1], label='Final Numerical')
plt.plot(r_array, history_ana[-1], '--', label="Final Analytical", alpha=0.8)
plt.axvline(x=r_fronts[-1], linestyle='--', color='gray', label="R_front")
plt.xlabel('Radial Distance r')
plt.ylabel('Temperature')
plt.title('Temperature Distribution (Final State)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Animation setup
fig, ax = plt.subplots(figsize=(10, 6),dpi=400)
line_num, = ax.plot([], [], lw=2, label='Numerical')
line_ana, = ax.plot([], [], lw=2, ls='--', label='Analytical')
line_step, = ax.plot([], [], lw=2, ls='-.', label='Step Function')
# line_front = ax.axvline(x=0, linestyle='--', color='gray', label='r_front(t*)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(r_array[0], r_array[-1])
ax.set_ylim(T_low, T_high + 0.1)
ax.set_xlabel('r*')
ax.set_ylabel('T*')
ax.legend()
ax.grid(True)

def init():
    line_num.set_data([], [])
    line_ana.set_data([], [])
    line_step.set_data([], [])
    # line_front.set_xdata([0, 0])
    time_text.set_text('')
    ax.set_title('')
    return line_num, line_ana, line_step, time_text

def animate(i):
    line_num.set_data(r_array, history[i])
    line_ana.set_data(r_array, history_ana[i])
    line_step.set_data(r_array, history_step[i])
    r_val = r_fronts[i]
    # line_front.set_xdata([r_val, r_val])
    time_text.set_text(f't* = {times[i]:.2f}')
    ax.set_title(f'Temperature Profile at t* = {times[i]:.2f}')
    return line_num, line_ana, line_step, time_text



ani = animation.FuncAnimation(fig, animate, frames=len(history),
                              init_func=init, blit=True, repeat=False)

# Save animation
output_path = "temperature_evolution_with_rfront.mp4"
ani.save(output_path, fps=20, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved to {output_path}")
