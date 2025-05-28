import cvxpy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

def fluid_value(T, N, lam, y0):
    """
    Solve the fluid model optimization problem.
    
    Parameters:
    - T: Time horizon
    - N: Scaling factor for control input
    - lam: Rate of change
    - y0: Initial state
    
    Returns:
    - u: Optimal control input
    """
    b = 2* N * np.sqrt(lam)  
    y_max   = y0 + T*lam 
    u = cp.Variable(T)
    y = cp.Variable(T+1)
    v = cp.Variable(T)  # Auxiliary variable for McCormick envelope

    # Objective function
    obj = cp.Minimize(cp.sum(N**2 * u + y[:-1]))

    # Constraints
    cons = [y[0] == y0,
            0 <= u, u <= 1,
            0 <= y,              # y ≥ 0
            0 <= v]              # first McCormick inequality

    # Dynamics with v instead of u*y
    for t in range(T):
        cons += [y[t+1] == y[t] + lam - v[t]]

    # Remaining McCormick envelope constraints
    cons += [
        v <= y[:-1],                         # v ≤ y_t
        v <= y_max * u,                      # v ≤ y_max u_t
        v >= y[:-1] - y_max * (1 - u)        # v ≥ y_t - y_max(1-u_t)
    ]

    prob = cp.Problem(obj, cons)
    prob.solve()   # LP; any LP/QP solver works

    return prob.value, u.value

def fluid_array(T, N, lam_values, upper_bound_capacity):
    #Solve the deterministic version of the multi-secretary problem for all periods in approx_periods and all capacities
    capacities = np.arange(10, upper_bound_capacity + 10, 10)
    val_deterministic = np.zeros((len(lam_values), capacities.shape[0]))
    solution_deterministic = np.zeros((len(lam_values), capacities.shape[0], T))
    for indx, lam in tqdm(enumerate(lam_values)):
        for indx_cap, x in tqdm(enumerate(capacities)):
            val_deterministic[indx, indx_cap] = fluid_value(T, N, lam, x)[0]
            solution_deterministic[indx, indx_cap, :] = fluid_value(T, N, lam, x)[1]

    return val_deterministic, solution_deterministic


if __name__ == "__main__":
    T = 500
    N = 10
    lam_values = np.array([0.2, 0.5, 0.9])
    upper_bound_capacity = 600

    val_deterministic, solution_deterministic = fluid_array(T, N, lam_values, upper_bound_capacity)

    print("Value Matrix:\n", val_deterministic)
    print("Solution Matrix:\n", solution_deterministic)

    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    x = np.arange(10, upper_bound_capacity + 10, 10)
    for i, lam in enumerate(lam_values):
        plt.plot(val_deterministic[i], label=f'λ = {lam}', linestyle=next(line_styles), marker='', fillstyle='none')
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Value function "$V(x0)+\gamma$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('x (State)', fontsize = 14)
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()

    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    capacities = np.arange(10, upper_bound_capacity + 10, 10)
    lam = .5
    for i, x in enumerate(capacities):
        if x in [50, 200, 300, 500, 600]:
            plt.plot(solution_deterministic[1, i][:50], label=f'x0 = {x}', linestyle=next(line_styles), marker='', fillstyle='none')
    plt.xlabel('t (Time)', fontsize = 14)
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Policy "$u(t)$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()