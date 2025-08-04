import cvxpy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import casadi as ca

def fluid_value(T, N, lam, y0):
    b = 2* N * np.sqrt(lam)  
    
    u = ca.MX.sym('u', T)
    y = ca.MX.sym('y', T+1)
    

    # objective
   
    J = 0
    for t in range(T):
        J += N**2*1/(1-u[t]) + y[t] 
    J = (J)/T

    # constraints list
    g = []
    # initial condition
    g.append(y[0] - y0)
    # dynamics
    for t in range(T):
        g.append(y[t+1] - (y[t] + lam - u[t]*y[t]))
        
    # pack into NLP
    w   = ca.vertcat(u, y)
    nlp = {'x': w, 'f': J, 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('S', 'ipopt', nlp, opts)

    # bounds
    lbg = [0]*(T+1)         # dynamics = 0, init = 0
    ubg = [0]*(T+1)
    lbx = [0]*T + [0]*(T+1)  # u >= 0, y >= 0
    ubx = [1]*T + [1e6]*(T+1)
    #u_fix = np.sqrt(lam)/N
    #lbx = [u_fix]*T     + [0]*(T+1) 
    #ubx = [u_fix]*T     + [1e6]*(T+1)

    # solve
    res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    w_opt = res['x'].full().flatten()
    u_opt = w_opt[:T]
    y_opt = w_opt[T:]
    return float(res['f']), u_opt, y_opt

def fluid_value_steady_state(T, N, lam, y0):
    b = 2* N * np.sqrt(lam)  
    
    u = ca.MX.sym('u', T)
    y = ca.MX.sym('y', T+1)
    

    # objective
   
    J = 0
    for t in range(T):
        J += N**2*u[t] + y[t] - b
    J = (J)/T

    # constraints list
    g = []
    # initial condition
    g.append(y[0] - y0)
    # dynamics
    for t in range(T):
        g.append(y[t+1] - y[t])
    for t in range(T):
        g.append(y[t+1] - (y[t] + lam - u[t]*y[t]))
    
        
    # pack into NLP
    w   = ca.vertcat(u, y)
    nlp = {'x': w, 'f': J, 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('S', 'ipopt', nlp, opts)

    # bounds
    lbg = [0]*len(g)
    ubg = [0]*len(g)
    lbx = [0]*T + [0]*(T+1)  # u >= 0, y >= 0
    ubx = [1]*T + [1e6]*(T+1)
    #u_fix = np.sqrt(lam)/N
    #lbx = [u_fix]*T     + [0]*(T+1) 
    #ubx = [u_fix]*T     + [1e6]*(T+1)

    # solve
    res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    w_opt = res['x'].full().flatten()
    u_opt = w_opt[:T]
    y_opt = w_opt[T:]
    return float(res['f']), u_opt, y_opt

def fluid_value_steady_control(T, N, lam, y0):
    b = 2* N * np.sqrt(lam)  
    
    u = ca.MX.sym('u', T)
    y = ca.MX.sym('y', T+1)
    

    # objective
   
    J = 0
    for t in range(T):
        J += N**2*u[t] + y[t] - b
    J = (J)/T

    # constraints list
    g = []
    # initial condition
    g.append(y[0] - y0)
    # dynamics
    for t in range(T):
        g.append(y[t+1] - (y[t] + lam - u[t]*y[t]))
        
    # pack into NLP
    w   = ca.vertcat(u, y)
    nlp = {'x': w, 'f': J, 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('S', 'ipopt', nlp, opts)

    # bounds
    lbg = [0]*(T+1)         # dynamics = 0, init = 0
    ubg = [0]*(T+1)
    u_fix = np.sqrt(lam)/N
    lbx = [u_fix]*T     + [0]*(T+1) 
    ubx = [u_fix]*T     + [1e6]*(T+1)

    # solve
    res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    w_opt = res['x'].full().flatten()
    u_opt = w_opt[:T]
    y_opt = w_opt[T:]
    return float(res['f']), u_opt, y_opt



def fluid_array(T, N, lam_values, upper_bound_capacity):
    #Solve the deterministic version of the multi-secretary problem for all periods in approx_periods and all capacities
    capacities = np.arange(0, upper_bound_capacity + 1, 1)
    val_deterministic = np.zeros((len(lam_values), capacities.shape[0]))
    solution_deterministic = np.zeros((len(lam_values), capacities.shape[0], T))
    states_deterministic = np.zeros((len(lam_values), capacities.shape[0], T+1))
    for indx, lam in enumerate(lam_values):
        for indx_cap, x in tqdm(enumerate(capacities)):
            val_deterministic[indx, indx_cap], solution_deterministic[indx, indx_cap, :], states_deterministic[indx, indx_cap, :] = fluid_value(T, N, lam, x)
            
    return val_deterministic, solution_deterministic, states_deterministic



if __name__ == "__main__":
    T = 2000
    N = 1
    lam_values = np.array([0.5]) 

    capacities = np.array([50])#np.array([np.sqrt(.5)*20])#np.arange(0, upper_bound_capacity + 1, 1)
    val_fluid = np.zeros((capacities.shape[0]))
    solution_fluid = np.zeros((capacities.shape[0], T))
    states_fluid = np.zeros((capacities.shape[0], T+1))
    val_steady_state = np.zeros((capacities.shape[0]))
    solution_steady_state = np.zeros((capacities.shape[0], T))
    states_steady_state = np.zeros((capacities.shape[0], T+1))
    val_steady_control = np.zeros((capacities.shape[0]))
    solution_steady_control = np.zeros((capacities.shape[0], T))
    states_steady_control = np.zeros((capacities.shape[0], T+1))

    for indx_cap, x in tqdm(enumerate(capacities)):
        val_fluid[indx_cap], solution_fluid[indx_cap, :], states_fluid[indx_cap, : ] = fluid_value(T, N, lam_values[0], x)
        val_steady_state[indx_cap], solution_steady_state[indx_cap, :], states_steady_state[indx_cap, : ] = fluid_value_steady_state(T, N, lam_values[0], x)
        val_steady_control[indx_cap], solution_steady_control[indx_cap, :], states_steady_control[indx_cap, : ] = fluid_value_steady_control(T, N, lam_values[0], x)

    line_styles = itertools.cycle(['--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(solution_fluid[0][:50], label=f'x0 =  N*sqrt(lamda)', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    #plt.plot(solution_steady_state[0], label=f'Steady State x_t = N*sqrt(lamda)', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    #plt.plot(solution_steady_control[0], label=f'Steady Control u = sqrt(lam)/N', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    plt.xlabel('t (Time)', fontsize = 14)
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Control "$u(t)$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
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
    plt.plot(states_fluid[0][:50], label=f'x0 =  N*sqrt(lamda)', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    #plt.plot(states_steady_state[0], label=f'Steady State x_t = N*sqrt(lamda)', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    #plt.plot(states_steady_control[0], label=f'Steady Control u = sqrt(lam)/N', linestyle=next(line_styles), marker='', fillstyle='none', color=next(colors))
    plt.xlabel('t (Time)', fontsize = 14)
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('States "$x(t)$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
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
    x=np.arange(0, upper_bound_capacity + 1, 1)
    for i, lam in enumerate(lam_values):
        plt.plot(x, val_deterministic[i], label=f'λ = {lam}', linestyle=next(line_styles), marker='', fillstyle='none')
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Value function "$V(x0)+\gamma$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('x0', fontsize = 14)
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()


    T = 1000
    N = 20
    lam_values = np.array([0.5]) 

    capacities = np.array([np.sqrt(.5)*20])#np.arange(0, upper_bound_capacity + 1, 1)
    val_fluid = np.zeros((capacities.shape[0]))
    solution_fluid = np.zeros((capacities.shape[0], T))
    states_fluid = np.zeros((capacities.shape[0], T+1))
    for indx_cap, x in tqdm(enumerate(capacities)):
        val_fluid[indx_cap], solution_fluid[indx_cap, :], states_fluid[indx_cap, : ] = fluid_value(T, N, lam_values[0], x)

    lam_values = np.array([0.5]) 
    upper_bound_capacity = 50
    capacities = np.arange(0, upper_bound_capacity + 1, 1)
    
    val_deterministic, solution_deterministic, states_deterministic = fluid_array(T, N, lam_values, upper_bound_capacity)



    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    x = np.arange(0, upper_bound_capacity + 1, 1)
    for i, lam in enumerate(lam_values):
        plt.plot(x, val_deterministic[i] - T*2* np.sqrt(lam)*N, label=f'λ = {lam}', linestyle=next(line_styles), marker='', fillstyle='none')
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Value function "$V(x0)+\gamma$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('x0', fontsize = 14)
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
    capacities = np.arange(0, upper_bound_capacity + 1, 1)
    lam = .5
    for i, x in enumerate(capacities):
        if x in [0, 10, 20, 30, 40, 50]:
            plt.plot(solution_deterministic[1, i][:200], label=f'x0 = {x}', linestyle=next(line_styles), marker='', fillstyle='none')
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

    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    lam = .5
    for i, x in enumerate(capacities):
        plt.plot(solution_fluid[i][:], label=f'x0 = {x}', linestyle=next(line_styles), marker='', fillstyle='none')
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

    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['black', 'red', 'blue'])
    plt.figure(figsize=(16,10), dpi= 80)
    for i, x in enumerate(capacities):
        plt.plot(states_fluid[i, : ], label=f'x0 = {x}', linestyle=next(line_styles), marker='', fillstyle='none')
    plt.xlabel('t (Time)', fontsize = 14)
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('States "$x(t)$"', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()


    ##########################################
    # Example for fluid_average_network
    T = 1000                # horizon length
    arrival_rates      = np.array([.1, .2, .5, .9],  dtype=float)          # arrival rates λ_i
    service_rates      = np.array([.18, .25, .70, .91], dtype=float)        # service rates μ_i
    rewards     = np.array([4, 3, 5, 2], dtype=float) # rewards r_i
    capacities = np.arange(1,10)                                          # capacity (Σ_i x_i ≤ q)
    initial_states = np.arange(2000)          
    # ------------------------------------------------------------
    average_costs_array = fluid_average_network_array(rewards, arrival_rates, service_rates, capacities)
    
    gamma = fluid_average_network(rewards, arrival_rates, service_rates, 10)  # average cost for capacity 10
    deterministic_value, deterministic_solution = fluid_value_network(T, rewards, arrival_rates, service_rates, 10, gamma, np.array([1,2,0,1]))

    # Plotting the average costs
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(capacities, average_costs_array, label='Average Costs', linestyle='-', marker='', fillstyle='none')
    plt.xlabel('Capacity', fontsize=14)
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Average Costs vs Capacity', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.gca().spines["top"].set_alpha(0.3)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()