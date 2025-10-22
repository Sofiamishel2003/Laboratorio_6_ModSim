import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, solve, pprint

# Valores de equilibrio
S_ele, I_ele = 1000, 0
S_end, I_end = 240, 380/3

# Creacion de variables
S, I, N= symbols('S I N')
beta, gamma, mu = symbols('beta gamma mu')

# Derivadas
dS_dt = mu*(N-S) - beta*S*I/N + gamma*I
dI_dt = beta*S*I/N - (mu + gamma)*I

# 1) Jacobiano
var_matrix = Matrix([S, I])
eq_matrix = Matrix([dS_dt, dI_dt])

jac = eq_matrix.jacobian(var_matrix)

print("# ==== Jacobian ==== #")
pprint(jac)

# 2) Eigen Valores (ELE)
print("\n# ==== Eigen Valores (ELE) ==== #")
jac_eq = jac.subs([
    (S, S_ele), (I, I_ele)
])
evals = jac_eq.eigenvals()
count = 1
for eigval, mult in evals.items():
    print(f"\tλ{count} = {eigval} (mult {mult})")
    count+=1

# 3) Equilibrios
print("\n# ==== Jacobiano con Parametros ==== #")
params = {
    N: 1000,
    beta: 0.5/1000,
    gamma: 0.1,
    mu: 0.02
}
jac_par = jac.subs(params)
pprint(jac_par)

print("\n# ==== Jacobiano ELE ==== #")
values = {
    S: S_ele, 
    I: I_ele
}
jac_ele = jac_par.subs(values)
pprint(jac_ele)

print("\n# ==== Jacobiano Endemico ==== #")
values = {
    S: S_end, 
    I: I_end
}
jac_end = jac_par.subs(values)
pprint(jac_end)



# 4) Numpy para eigenvalores
print("\n# ==== Eigenvalores Equilibrio Endemico ==== #")
jac_end_np = np.array(jac_end).astype(np.float64)
end_eigvals = np.linalg.eigvals(jac_end_np)
print(end_eigvals)