import numpy as np
import matplotlib.pyplot as plt

def solve_pressure_equation(r0, r1, N, p0, dp0):
    h = (r1 - r0) / N
    r = np.linspace(r0, r1, N + 1)
    p = np.zeros(N + 1)

    # Initial conditions
    p[0] = p0
    p[1] = p0 + h * dp0  # Forward Euler to estimate p[1]

    for i in range(1, N):
        p[i + 1] = -(-2 + h/(r0 + i*h)) * p[i] - (1 - h/(r0 + i*h)) * p[i - 1]

    return r, p

r_w = 1 # radius of well
r_out = r_w + 5

kappa = 1e3 # k/mu
u_inj = 10 # Darcy velocity of injection
dp_w = -1/kappa * u_inj
p_w = 10 # pressure at rw, which can be assigned arbitrarily

# Solve and plot
r, p = solve_pressure_equation(r_w, r_out, N=100, p0=p_w, dp0=dp_w)

c2 = 1/kappa * r_w * u_inj * np.log(r_w) + p_w
p_ex = - 1/kappa * r_w * u_inj * np.log(r) + c2

plt.plot(r, p, label='Numerical solution')
plt.plot(r, p_ex, label='Exact solution')
plt.xlabel('r')
plt.ylabel('p(r)')
plt.title('Solution of r d²p/dr² + dp/dr = 0')
plt.grid()
plt.legend()
plt.show()