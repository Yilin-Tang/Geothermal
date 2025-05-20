import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import os

phi = 0.15 # porosity
D = 1.5e-9 # diffusion coefficient
Peclet = 4.2
gamma = 1.67
T_low = 20
rp_w = 0.05
up_inj = 0.05

# reaction parameters
R = 8.31446261815324 # universal gas constant
A1 = 3.55e3
A2 = 1.21
A3 = -1.33e5
A4 = -1.41e3
A5 = 5.08e6
A6 = -4.55e-4
A = np.ones(4)*4.65e-2 # Arrhenius pre-exponential factor
Ea = np.ones(4)*34000
eta0 = 2.0
theta0 = 0.5
v = np.ones(4)*1.0
SA = np.ones(4)*1.0
act = [1.262e-1, 2.575e-3, 4.691e-5, 1.658e-2]
IAP = np.zeros(4)
IAP[0] = act[0] * act[1] # NaCl
IAP[1] = act[0]**2 * act[3] # Na2SO4
IAP[2] = act[2] * act[1]**2 # SrCl2
IAP[3] = act[2] * act[3] # SrSO4

C0 = np.array([1.832e-1, 3.912e-3, 1.946e-4, 7.978e-2]) # Na, Cl, Sr, SO4
C_inj = np.array([1.84e-1, 4.0e-3, 1.0e-2, 1.0e-1])

def reaction(C, T):
    num_species = C.shape[0]
    K_diss = np.zeros(C.shape)
    K_eq = np.zeros(C.shape)
    S = np.zeros(C.shape)

    for j in range(num_species):
        K_diss[j] = 4.8e-7 #A[j] * np.exp(-Ea[j]/(R*T))
        K_eq[j] = A1 + A2*T + A3/T + A4*np.log10(T) + A5*(T**2) + A6*T**2
        K_eq[j] = np.power(K_eq[j], 10)

    def single_reaction(i):
        return v[i] * SA[i] * K_diss[i,:]*np.power(np.power(IAP[i] / K_eq[i,:], theta0)- 1.0,eta0)
    
    S[0,:] = single_reaction(0) + single_reaction(1)
    S[1,:] = single_reaction(0) + single_reaction(2)
    S[2,:] = single_reaction(2) + single_reaction(3)
    S[3,:] = single_reaction(1) + single_reaction(3)

    return S

def solve_concentration_equation(r0, r1, delta_r, time, delta_t, num_species, T_ic, T_bc, C_ic, C_bc):
    Nr = int((r1 - r0) / delta_r)
    r = np.linspace(r0, r1, Nr + 1)
    Nt = int(time / delta_t)
    t = np.linspace(0, time, Nt + 1)
    T = T_ic * np.ones(Nr + 1)
    T[0] = T_bc
    C = np.tile(C_ic.reshape(-1, 1), (1, Nr + 1))
    C[:,0] = C_bc


    Ta_array = delta_t / (Peclet * r * delta_r)         # appears in the T[i+1] term
    Tb_array = delta_t / (Peclet * delta_r**2) * np.ones(Nr + 1)      # appears in the T[i-1] term
    Tc_array = -gamma * delta_t / (r * delta_r)            # additional term for T[i+1]


    Tmain_diag = 1 + Ta_array[1:-1] - 2 * Tb_array[1:-1] + Tc_array[1:-1]
    Tlower_diag = -Ta_array[2:-1] + Tb_array[2:-1] - Tc_array[2:-1]                    # length = n_interior - 1
    Tupper_diag = Tb_array[1:-2]  # length = n_interior - 1

    TA = diags([Tlower_diag, Tmain_diag, Tupper_diag], offsets=[-1, 0, 1], format='csc')

    Ca_array = delta_t*D / (phi * r * delta_r)         # appears in the C[i+1] term
    Cb_array = delta_t*D / (phi * delta_r**2) * np.ones(Nr + 1)      # appears in the C[i-1] term
    Cc_array = -rp_w*up_inj / phi * delta_t / (r * delta_r)            # additional term for C[i+1]

    Cmain_diag = 1 + Ca_array[1:-1] - 2 * Cb_array[1:-1] + Cc_array[1:-1]
    Clower_diag = -Ca_array[2:-1] + Cb_array[2:-1] - Cc_array[2:-1]                     # length = n_interior - 1
    Cupper_diag = Cb_array[1:-2]  # length = n_interior - 1

    CA = diags([Clower_diag, Cmain_diag, Cupper_diag], offsets=[-1, 0, 1], format='csc')

    T_history = [T.copy()]
    C_history = [C.copy()]

    for it in range(Nt):
        C[:, 0] = C_bc         # enforce inner Dirichlet BC
        C[:, -1] = C[:, -2]        # enforce outer Neumann BC
        C_interior = C[:, 1:-1].copy()
        T_interior = T[1:-1].copy()
        C_interior_new = np.array([CA @ C_interior[i, :] for i in range(num_species)]) - delta_t * reaction(C_interior, T_interior) 
        React = reaction(C_interior, T_interior)
        # add boundary contributions:
        C_interior_new[:, 0] += (-Ca_array[1] + Cb_array[1] - Cc_array[1]) * C[:, 0]
        C_interior_new[:, -1] += (Cb_array[-2]) * C[:, -1]

        C[:,1:-1] = C_interior_new

        T[0] = T_bc         # enforce inner Dirichlet BC
        T[-1] = T[-2]        # enforce outer Neumann BC
        T_interior_new = TA.dot(T_interior)

        # add boundary contributions:
        T_interior_new[0] += (-Ta_array[1] + Tb_array[1] - Tc_array[1]) * T[0]
        T_interior_new[-1] += (Tb_array[-2]) * T[-1]

        T[1:-1] = T_interior_new

        if it % (ITER // 100) == 0:
            C_history.append(C.copy())
            T_history.append(T.copy())


    return r, t, T, C, T_history, C_history

Tp_high = 80
Tp_low = 20

rp_out = 5
delta_r = 0.1
delta_t = 1e-2

ITER = 10000
final_time_p = ITER * delta_t
print(f"Final time: {final_time_p:.2f} seconds")
r, t, T, C, T_history, C_history = solve_concentration_equation(rp_w, rp_out, delta_r, final_time_p, delta_t, 4, Tp_high, Tp_low, C0, C_inj)

plt.figure(figsize=(10,6))
plt.plot(r, T_history[-1], label='Final Temperature')
plt.xlabel('Radial Distance r')
plt.ylabel('Temperature')
plt.title('Temperature Distribution (Final State)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(r, C_history[-1][i,:], label=f'Line {i+1}') 

plt.xlabel('Radial Distance r')
plt.ylabel('Concentration')
plt.title('Concentration Distribution (Final State)')
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f'Line {j+1}')[0] for j in range(4)]
ax.set_xlim(r.min(), r.max())
ax.set_ylim(0, 0.2)
ax.set_xlabel('Radial Distance r')
ax.set_ylabel('Concentration')
ax.set_title('Concentration Evolution Over Time')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(i):
    for j, line in enumerate(lines):
        line.set_data(r, C_history[i][j])
    ax.set_title(f'Concentration Evolution - Time Step {i+1}/{len(C_history)}')
    return lines

ani = animation.FuncAnimation(fig, animate, frames=len(C_history), init_func=init,
                              blit=True, repeat=False)

# Save the animation as MP4
output_path = "concentration_evolution.mp4"
ani.save(output_path, fps=10, extra_args=['-vcodec', 'libx264'])