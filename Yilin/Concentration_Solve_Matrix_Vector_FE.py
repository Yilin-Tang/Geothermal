import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
import scipy 
import os

import scipy.special

phi = 0.2 # porosity
# phi = 0.15 # porosity
D = 1.5e-7 # diffusion coefficient
c_w = 4.18e3
rho_w = 1000
c_s = 0.8e3
rho_s = 2600
r_w = 0.05
r_out = 25
u_inj = 3.17e-5 #0.00186
#u_inj = 0.001 #0.00186
alpha = 6.8e-7
Peclet = (u_inj * r_w) / alpha
Pe0 = u_inj*r_w/D
gamma = c_w * rho_w / (c_w * rho_w * phi + c_s * rho_s * (1 - phi))
#Peclet = 4.66
#gamma = 1.672

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
#eta0 = 2.0
theta0 = 0.1
v = np.ones(4)*1.0
SA = np.ones(4)*1.0
act = [1.262e-1, 2.575e-3, 4.691e-5, 1.658e-2]
IAP = np.zeros(4)
IAP[0] = act[0] * act[1] # NaCl
IAP[1] = act[0]**2 * act[3] # Na2SO4
IAP[2] = act[2] * act[1]**2 # SrCl2
IAP[3] = act[2] * act[3] # SrSO4

#C0 = np.array([1.832e-1, 3.912e-3, 1.946e-4, 7.978e-2]) # Na, Cl, Sr, SO4
#C_inj = np.array([1.84e-1, 4.0e-3, 1.0e-2, 1.0e-1])
C0 = np.array([1.946e-4]) #Sr, SO4
C_inj =  np.array([1.0e-2])

def anal_conc(x, t, Pe0, phi):
    v = np.sqrt(1+2*t/phi)
    eps = 1 / Pe0
    s = t**2 / phi + t
    #print(f't:{t}, s:{s}')
    xi = (x-v) / np.sqrt(eps)
    d = xi * v

    return (1 - scipy.special.erf((np.sqrt(phi) * d) / (2*np.sqrt(s)))) * 0.5   

def anal_temp(x, t, Pe, gamma):
    v = np.sqrt(1+2*t*gamma)
    eps = 1 / Pe
    s = t**2 *gamma + t
    #print(f't:{t}, s:{s}')
    xi = (x-v) / np.sqrt(eps)
    d = xi * v

    return (1 + scipy.special.erf((d) / (2*np.sqrt(s)))) * 0.5   

def simplified_reaction(C, T):
   S = np.zeros(C.shape)
   K_diss = A[0] * np.exp(-Ea[0]/(R*(80+273.15))) 
#    K_diss = A[0] * np.exp(-Ea[0]/(R*(T+273.15))) #transform ℃ to K
   #print(np.max(K_diss))
   S[0,:] = SA[0] * K_diss * (C/C0[0] - 1.0)
   return S

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


    Ta_array = delta_t*alpha / (r * delta_r)         # appears in the T[i+1] term
    Tb_array = delta_t*alpha / (delta_r**2) * np.ones(Nr + 1)      # appears in the T[i-1] term
    Tc_array = -r_w*u_inj * gamma * delta_t / (r * delta_r)            # additional term for T[i+1]


    Tmain_diag = 1 + Ta_array[1:-1] - 2 * Tb_array[1:-1] + Tc_array[1:-1]
    Tlower_diag = -Ta_array[2:-1] + Tb_array[2:-1] - Tc_array[2:-1]                    # length = n_interior - 1
    Tupper_diag = Tb_array[1:-2]  # length = n_interior - 1

    # Tmain_diag = 1 - Ta_array[1:-1] - 2 * Tb_array[1:-1] - Tc_array[1:-1]
    # Tlower_diag = Tb_array[2:-1]                   
    # Tupper_diag = Ta_array[1:-2] + Tb_array[1:-2] + Tc_array[1:-2]

    TA = diags([Tlower_diag, Tmain_diag, Tupper_diag], offsets=[-1, 0, 1], format='csc')

    Ca_array = delta_t*D / (phi * r * delta_r)         # appears in the C[i+1] term
    Cb_array = delta_t*D / (phi * delta_r**2) * np.ones(Nr + 1)      # appears in the C[i-1] term
    Cc_array = (-r_w*u_inj / phi) * (delta_t / (r * delta_r))            # additional term for C[i+1]

    # (c_i - c_i-1)/delta_r) stable
    Cmain_diag = 1 + Ca_array[1:-1] - 2 * Cb_array[1:-1] + Cc_array[1:-1]
    Clower_diag = -Ca_array[2:-1] + Cb_array[2:-1] - Cc_array[2:-1]                     # length = n_interior - 1
    Cupper_diag = Cb_array[1:-2]  # length = n_interior - 1

    # (c_i+1 - c_i)/delta_r)
    # Cmain_diag = 1 - Ca_array[1:-1] - 2 * Cb_array[1:-1] - Cc_array[1:-1]
    # Clower_diag = Cb_array[2:-1]                   
    # Cupper_diag = Ca_array[1:-2] + Cb_array[1:-2] + Cc_array[1:-2]

    CA = diags([Clower_diag, Cmain_diag, Cupper_diag], offsets=[-1, 0, 1], format='csc')

    # T_history = [T[:len(T)//5].copy()]
    # C_history = [C[:, :len(T)//5].copy()]
    # nondim_con = anal_conc(r/r_w, time * u_inj / r_w, u_inj*r_w/D, phi)
    # dim_anal_con = (C_inj[0] - C0[0]) * nondim_con + C0[0]
    # C_history_anal = [dim_anal_con[:len(T)//5]]
    T_history = []
    C_history = []
    C_history_anal = []
    
    for it in range(Nt):
        C[:, 0] = C_bc         # enforce inner Dirichlet BC
        C[:, -1] = C[:, -2]        # enforce outer Neumann BC
        T_interior = T[1:-1].copy()
        C_interior = C[:, 1:-1].copy()
        C_interior_new = np.array([CA @ C_interior[i, :] for i in range(num_species)]) - delta_t * simplified_reaction(C_interior, T_interior) 
        # add boundary contributions:
        C_interior_new[:, 0] += (-Ca_array[1] + Cb_array[1] - Cc_array[1]) * C[:, 0]
        C_interior_new[:, -1] += (Cb_array[-2]) * C[:, -1]
        # C_interior_new[:, 0] += Cb_array[1] * C[:, 0]
        # C_interior_new[:, -1] += (Ca_array[-2] + Cb_array[-2] + Cc_array[-2]) * C[:, -1]

        C[:,1:-1] = C_interior_new

        T[0] = T_bc         # enforce inner Dirichlet BC
        T[-1] = T[-2]        # enforce outer Neumann BC
        T_interior_new = TA.dot(T_interior)

        # add boundary contributions:
        T_interior_new[0] += (-Ta_array[1] + Tb_array[1] - Tc_array[1]) * T[0]
        T_interior_new[-1] += (Tb_array[-2]) * T[-1]
        # T_interior_new[0] += Tb_array[1] * T[0]
        # T_interior_new[-1] += (Ta_array[-2] + Tb_array[-2] + Tc_array[-2]) * T[-1]

        T[1:-1] = T_interior_new

        if it % (ITER // 100) == 0:
            nondim_con = anal_conc(r/r_w, it*delta_t * u_inj / r_w, u_inj*r_w/D, phi)
            dim_anal_con = (C_inj[0] - C0[0]) * nondim_con + C0[0]
            C_history_anal.append(dim_anal_con[:len(T)//5])
            C_history.append(C[:, :len(T)//5].copy())
            T_history.append(T[:len(T)//5].copy())

    return r, t, T, C, T_history, C_history, C_history_anal


Tp_high = 80
Tp_low = 20

delta_r = 0.001
delta_t = 0.5

ITER = 200000
final_time_p = ITER * delta_t
print(f"Final time: {final_time_p:.2f} seconds")
r, t, T, C, T_history, C_history, C_history_anal = solve_concentration_equation(r_w, r_out, delta_r, final_time_p, delta_t, 1, Tp_high, Tp_low, C0, C_inj)

indices = [0, 1, 3, 9, 20, 30, 40, 99]
indices = [0, 1, 9, 25, 45, 99]
num_line = 8

plt.figure(figsize=(10,6))
for j in indices[1:]:
    time = j*delta_t*ITER/100
    plt.plot(r[:len(r)//5], T_history[j], label=f'time {time}s') 
    nondim_temp = anal_temp(r/r_w, time * u_inj / r_w, Peclet, gamma)
    dim_anal_conc = (Tp_high - Tp_low) * nondim_temp + Tp_low
    plt.plot(r[:len(r)//5], dim_anal_conc[:len(r)//5], linestyle='--', label=f'anal at t={time}s')

plt.xlabel('Radial Distance r')
plt.ylabel('Temperature')
plt.title('Temperature Distribution (Final State)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

colormap = plt.colormaps['viridis']
color_values = np.linspace(0, 1, num_line*2)
colors = [colormap(val) for val in color_values]

print(indices)
index = 0
plt.figure(figsize=(10,6))
for i in range(1):
    for j in indices[1:]:
        time = j*delta_t*ITER/100
        # nondim_con = anal_conc(r/r_w, time * u_inj / r_w, u_inj*r_w/D, phi)
        # dim_anal_conc = (C_inj[0] - C0[0]) * nondim_con + C0[0]
        # plt.plot(r[:len(r)//5], C_history_anal[j], color=colors[index], linestyle='--', label=f'anal at t={time}s')
        plt.plot(r[:len(r)//5], C_history[j][i,:], color=colors[index], linestyle='-', label=f'num at t={time}s') 
        index += 3

plt.xlabel('Radial Distance r')
plt.ylabel('Concentration')
plt.title(f'Concentration Distribution at Different Times; Peclet:{Pe0:.2f}')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

"""
fig, ax = plt.subplots(figsize=(10, 6))
line_num, = ax.plot([], [], lw=2, label='Numerical')
line_anal, = ax.plot([], [], lw=2, ls='--', label='Analytical')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(r.min(), r[len(r)//5])
ax.set_ylim(0, 0.01)
ax.set_xlabel('Radial Distance r')
ax.set_ylabel('Concentration')
ax.legend()
ax.set_title('Concentration Evolution Over Time')
ax.grid(True)

def init():
    line_num.set_data([], [])
    #line_anal.set_data([], [])
    time_text.set_text('')
    ax.set_title('')
    return line_num, line_anal, time_text

def animate(i):
    line_num.set_data(r[:len(r)//5], C_history[i][0,:])
    line_anal.set_data(r[:len(r)//5], C_history_anal[i])
    time_text.set_text(f't = {t[i]*ITER/100:.2f}')
    ax.set_title(f'Concentration Profile at t = {t[i]*ITER/100:.2f}')
    return line_num, line_anal, time_text
    return line_num, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(C_history), init_func=init, blit=True, repeat=False)
# Save the animation as MP4
output_path = "concentration_num_anal_evolution.mp4"
ani.save(output_path, fps=10, extra_args=['-vcodec', 'libx264'])

fig, ax = plt.subplots(figsize=(10, 6))
line_num, = ax.plot([], [], lw=2, label='Numerical')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(r.min(), r[len(r)//5])
ax.set_ylim(0, 0.01)
ax.set_xlabel('Radial Distance r')
ax.set_ylabel('Concentration')
ax.legend()
ax.set_title('Concentration Evolution Over Time')
ax.grid(True)

def init2():
    line_num.set_data([], [])
    #line_anal.set_data([], [])
    time_text.set_text('')
    ax.set_title('')
    return line_num, time_text

def animate(i):
    line_num.set_data(r[:len(r)//5], C_history[i][0,:])
    # line_anal.set_data(r[:len(r)//5], C_history_anal[i])
    time_text.set_text(f't = {t[i]*ITER/100:.2f}')
    ax.set_title(f'Concentration Profile at t = {t[i]*ITER/100:.2f}')
    #return line_num, line_anal, time_text
    return line_num, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(C_history), init_func=init2, blit=True, repeat=False)
# Save the animation as MP4
output_path = "concentration_num_evolution.mp4"
ani.save(output_path, fps=10, extra_args=['-vcodec', 'libx264'])
"""