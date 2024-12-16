import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os 

#############################################################
#####scheduling single server 
############################################################

def fluid_queue(probabilities, costs, T, x):
    total_cost = 0
    for t in range(0, T+1):
        x_1_t = np.maximum(0,x[0] - probabilities[0]*t)
        if t<= x[0]/probabilities[0]:
            x_2_t = x[1]
        else:
            x_2_t = np.maximum(0,x[1] - probabilities[1]*(t - x[0]/probabilities[0]))
        x_t = np.array([x_1_t, x_2_t])
        cost = np.dot(costs, x_t)
        total_cost += cost
    return total_cost

def fluid_queue_array(probabilities, costs, T, X):
    val_fluid = np.zeros((T+1, X+1, X+1))
    for t in tqdm(range(0, T+1)):
        for x1 in range(0, X+1):
            for x2 in range(0, X+1):
                cost = fluid_queue(probabilities, costs, t, np.array([x1, x2]))
                val_fluid[t][x1][x2] = cost
    return val_fluid

def dynamic_queue(T, X, probabilities, costs):
    val = np.zeros((T+1, X+1, X+1))
    sol = np.zeros((T+1, X+1, X+1), dtype=int)
    
    for t in tqdm(np.arange(0, T+1)):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    val[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                elif x1 == 0:
                    if x2==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_2 = val[t-1][x1][x2-1]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[1]*x2
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                elif x2 == 0:
                    if x1==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_1 = val[t-1][x1-1][x2]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[0]*x1
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:   
                    next_less_1 = val[t-1][x1-1][x2]
                    next_less_2 = val[t-1][x1][x2-1]
                    next_same = val[t-1][x1][x2]
                    current_cost = np.dot(costs, np.array([x1, x2]))

                    logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same <= current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                    if logic_test:
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                        sol[t][x1][x2] = 1
                    else:
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                        sol[t][x1][x2] = 0

    return val, sol

def dynamic_lookahead(T, X, val_deterministic, window_size, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    solution = np.zeros((T+1, X+1, X+1), dtype=int)

    for period in tqdm(range(0, T+1)): 
        val_temp = np.zeros((T+1, X+1, X+1))
        sol_temp = np.zeros((T+1, X+1, X+1), dtype=int)
        window = int(np.ceil(period**(window_size))) + 1
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                val_temp[0][x1][x2] = np.dot(costs, np.array([x1, x2]))
        small_t = period-window+1 if period-window > 0 else 1
        for t in np.arange(small_t, period+1):
            for x1 in np.arange(0, X+1):
                for x2 in np.arange(0, X+1):
                    if t == 0:
                        val_temp[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                    elif x1 == 0:
                        if x2==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_2 = val_deterministic[t-1][x1][x2-1]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_2 = val_temp[t-1][x1][x2-1]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[1]*x2
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            #sol_temp[t][x1][x2] = 0
                    elif x2 == 0:
                        if x1==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_1 = val_deterministic[t-1][x1-1][x2]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_1 = val_temp[t-1][x1-1][x2]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[0]*x1
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            #sol_temp[t][x1][x2] = 1
                    else:
                        if t == period-window+1:
                            next_less_1 = val_deterministic[t-1][x1-1][x2]
                            next_less_2 = val_deterministic[t-1][x1][x2-1]
                            next_same = val_deterministic[t-1][x1][x2]
                        else: 
                            next_less_1 = val_temp[t-1][x1-1][x2]
                            next_less_2 = val_temp[t-1][x1][x2-1]
                            next_same = val_temp[t-1][x1][x2]
                        
                        current_cost = np.dot(costs, np.array([x1, x2]))

                        logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same < current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                        
                        if logic_test:
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                        else:
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol_temp[t][x1][x2] = 0

                    if t == period:
                        value[t][x1][x2] = val_temp[t][x1][x2]
                        solution[t][x1][x2] = sol_temp[t][x1][x2]

    return value, solution

def dynamic_evaluate_solution(T, X, sol, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    
    for t in np.arange(0, T+1):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    value[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                elif x1 == 0:
                    if x2==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_2 = value[t-1][x1][x2-1]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[1]*x2
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            
                elif x2 == 0:
                    if x1==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_1 = value[t-1][x1-1][x2]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[0]*x1
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:        
                    next_less_1 = value[t-1][x1-1][x2]
                    next_less_2 = value[t-1][x1][x2-1]
                    next_same = value[t-1][x1][x2]
                    current_cost = np.dot(costs, np.array([x1, x2]))
                    if sol[t][x1][x2] == 1:
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                    else:
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same        

    return value

############################################################################
############################################################################
##  Main code
############################################################################
############################################################################
probabilities = np.array([0.5, 0.5])
costs = np.array([8, 2])
T = 100
X = 100
window_size = 4/5

val_fluid_queue = fluid_queue_array(probabilities, costs, T, X)
val_dp_queue, sol_dp_queue = dynamic_queue(T, X, probabilities, costs)
val_lookahead, sol_lookahead = dynamic_lookahead(T, X, val_fluid_queue, window_size, probabilities, costs)
val_eval_lookahead = dynamic_evaluate_solution(T, X, sol_lookahead, probabilities, costs)
sub_opt_gap = np.zeros(T+1)
for t in range(0, T+1):
    sub_opt_gap[t] = np.max(val_eval_lookahead[t]-val_dp_queue[t])
 

############################################################################
############################################################################
#Plots for Presentation
############################################################################
############################################################################

path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'
path_presentation = path_0 + 'Figures/presentation/'
if not os.path.exists(path_presentation):
    os.makedirs(path_presentation)

#############
## Regret
##############
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sub_opt_gap, color = 'tab:blue', label='Regret', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_regret.png')
plt.close()

############
#Plot of value functions surface
############
t = 100
X = 100 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue[t, 0:X+1, 0:X+1] 
z2 = val_dp_queue[t, 0:X+1, 0:X+1]
#z3 = val_eval_lookahead[t, 0:X+1, 0:X+1]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
#surf3 = ax.plot_surface(x1, x2, z3, cmap='Grays')
# Add labels and title
ax.set_xlabel('K2')
ax.set_ylabel('K1')
ax.set_zlabel('Cost')
#ax.set_title(f'Surface Plot of value functions at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


#############
#Plot of difference between value functions 
#############
z = val_eval_lookahead[t, 0:X+1, 0:X+1]-val_fluid_queue[t, 0:X+1, 0:X+1]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='viridis')
# Add labels and title
ax.set_xlabel('K2')
ax.set_ylabel('K1')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


#################
## Plots of derivatives of value functions
#################
t=100
k1 = 30
derivative_x_2_fluid = np.diff(val_fluid_queue[t], n=1, axis=1)
derivative_x_2_dp = np.diff(val_dp_queue[t], n=1, axis=1)
derivative_x_2_lookahead = np.diff(val_lookahead[t], n=1, axis=1)
derivative_x_2_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 1)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_type2.png')
plt.close()


plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_2_lookahead[k1], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_lookahead_type2.png')
plt.close()

k2 = 30
derivative_x_1_fluid = np.diff(val_fluid_queue[t], n=1, axis=0)
derivative_x_1_dp = np.diff(val_dp_queue[t], n=1, axis=0)
derivative_x_1_lookahead = np.diff(val_lookahead[t], n=1, axis=0)
derivative_x_1_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 0)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_type1.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_1_lookahead[:,k2], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_lookahead_type1.png')
plt.close()


t = 50
derivative_x_2 = np.diff(val_dp_queue[t], n=1,axis=1)
derivative_x_2_fluid = np.diff(val_fluid_queue[t], n=1,axis=1)
# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_2[1:,:]  # Slice to exclude zeros for x1 and x2
z2 = derivative_x_2_fluid[1:,:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
# Add labels and title
ax.set_xlabel('x2')
ax.set_ylabel('x1')
ax.set_zlabel('Difference Cost')
ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


derivative_x_1 = np.diff(val_dp_queue[t], n=1,axis=0)
derivative_x_1_fluid = np.diff(val_fluid_queue[t], n=1,axis=0)
# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_1[:,1:]  # Slice to exclude zeros for x1 and x2
z2 = derivative_x_1_fluid[:,1:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
# Add labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Difference Cost')
ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
