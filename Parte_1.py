import numpy as np
import matplotlib.pyplot as plt

# Modelo SIR con dinámica vital (nacimiento/muerte) pero petit pretty
def sir_vital_deriv(state, t, beta, gamma, mu, N):
    S, I = state
    dS = mu*N - beta*S*I - mu*S
    dI = beta*S*I - (gamma + mu)*I
    return np.array([dS, dI])

# Runge-Kutta 4th order integrator
def rk4(f, y0, t, args=()):
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for k in range(1, len(t)):
        h = t[k] - t[k-1]
        tk = t[k-1]
        yk = y[k-1]
        k1 = f(yk, tk, *args)
        k2 = f(yk + 0.5*h*k1, tk + 0.5*h, *args)
        k3 = f(yk + 0.5*h*k2, tk + 0.5*h, *args)
        k4 = f(yk + h*k3, tk + h, *args)
        y[k] = yk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# Parametros del modelo según las instrucciones de mi amigo Suriano
N = 1000
beta = 0.5 / N     # transmisión
gamma = 0.1        # recuperación
mu = 0.02          # nacimiento/muerte
R0 = (beta * N) / (gamma + mu)
T = 300.0
dt = 0.1
t = np.arange(0.0, T + dt, dt)

# Condiciones iniciales (2 escenarios)
ICs = {
    "Inicio de brote (S0=999, I0=1)": np.array([999.0, 1.0]),
    "Epidemia mayor (S0=700, I0=300)": np.array([700.0, 300.0])
}

# Simulación de trayectorias
trajectories = {}
for label, y0 in ICs.items():
    traj = rk4(sir_vital_deriv, y0, t, args=(beta, gamma, mu, N))
    traj = np.clip(traj, a_min=0.0, a_max=N)
    trajectories[label] = traj

# Libre de enferemedad(ELE): I* = 0, S* = N
S_ele = N
I_ele = 0.0

# Endémico: S* = (gamma + mu)/beta, I* = (mu/(gamma + mu))*(N - S*)
S_end = (gamma + mu) / beta
I_end = (mu / (gamma + mu)) * (N - S_end)

# Impresión de resultados
print("Parámetros:")
print(f"  N = {N}, beta = {beta:.6f}, gamma = {gamma}, mu = {mu}")
print(f"  R0 = {R0:.4f} (> 1)\n")
print("Puntos de equilibrio:")
print(f"  ELE (Libre de enfermedad): S* = {S_ele:.4f}, I* = {I_ele:.4f}")
print(f"  Endémico:                  S* = {S_end:.4f}, I* = {I_end:.4f}")

# Grafica de las trayectorias en el espacio de estados (S vs I)
plt.figure(figsize=(8,6))
for label, traj in trajectories.items():
    S, I = traj[:,0], traj[:,1]
    plt.plot(S, I, label=label)
# Marcar equilibrios
plt.scatter([S_ele, S_end], [I_ele, I_end], s=60, marker='o')
plt.annotate("ELE", (S_ele, I_ele), xytext=(S_ele-120, I_ele+25), arrowprops=dict(arrowstyle="->"))
plt.annotate("Endémico", (S_end, I_end), xytext=(S_end-250, I_end+40), arrowprops=dict(arrowstyle="->"))
plt.xlabel("Número de Susceptibles (S)")
plt.ylabel("Número de Infectados (I)")
plt.title("Modelo SIR con dinámica vital: espacio de estados (S vs I)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
