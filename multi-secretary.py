import numpy as np 
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import uniform

###############################################################
# Dynamic programing approach to multisecreatry problem with n 
# types, only one type i apearing each time step with prob p_i
###############################################################
def generate_vectors(n):
    # Create a list of [0, 1] repeated n times
    arrays = [np.array([0, 1])] * n
    # Use np.meshgrid to create the grid of combinations
    grid = np.meshgrid(*arrays)
    # Reshape the grid to get the desired combinations
    vectors = np.array(grid).T.reshape(-1, n)
    return vectors

def deterministic_msecretary(probabilities, rewards, n_types, t, x):
    y = cp.Variable(n_types)
    objective = cp.Maximize(cp.sum(rewards @ y))
    constraints = [0<=y, y<= probabilities*t, cp.sum(y)<=x]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return result

def msecretary(val, sol_index, prob_choice, rewards, t, x): 
    if t == 0 or x == 0:
        return 0 
    if val[t][x] != 0: 
        return val[t][x]
    
    q_val = (np.sum(prob_choice * (rewards + msecretary(val, sol_index, prob_choice, rewards, t-1, x-1)), axis = 1) 
             + (1-prob_choice.sum(axis = 1))*msecretary(val, sol_index, prob_choice, rewards, t-1, x))
    
    sol_index[t][x] = np.argmax(q_val)
    val[t][x] = q_val[sol_index[t][x]]
    
    return val[t][x]

def dynamic_solution(T, capacity):
    
    val = np.zeros((T+1, capacity+1))
    sol_index = np.zeros((T+1, capacity+1), dtype=int)

    result = np.zeros((capacity+1))
    for cap in range(1, capacity+1):
        result[cap] = msecretary(val, sol_index, prob_choice, rewards, T, cap)
    
    sol = vectors[sol_index]

    return result, val, sol, sol_index

def approx_dynamic_solution(T, capacity):

    val = np.zeros((T+1, capacity+1))
    sol_index = np.zeros((T+1, capacity+1), dtype=int)

    for t in range(1, T+1):
        for x in range(1, capacity+1):
            if t == 0 or x == 0:
                val[t][x] = 0 
    
            q_val = (np.sum(prob_choice * (rewards + deterministic_msecretary(probabilities, rewards, n_types, t-1, x-1)), axis = 1) 
                     + (1-prob_choice.sum(axis = 1))*deterministic_msecretary(probabilities, rewards, n_types, t-1, x))
            sol_index[t][x] = np.argmax(q_val)
            val[t][x] = q_val[sol_index[t][x]]

    result = val[T][capacity]
    sol = vectors[sol_index]

    return result, val, sol, sol_index

def evaluate_msecretary(val, sol_index, prob_choice, rewards, t, x): 
    if t == 0 or x == 0:
        return 0 
    if val[t][x] != 0: 
        return val[t][x]
    
    q_val = (np.sum(prob_choice * (rewards + evaluate_msecretary(val, sol_index, prob_choice, rewards, t-1, x-1)), axis = 1) 
             + (1-prob_choice.sum(axis = 1))*evaluate_msecretary(val, sol_index, prob_choice, rewards, t-1, x))
    
    val[t][x] = q_val[sol_index[t][x]]
    
    return val[t][x]

def evaluate_solution(T, capacity, sol_index):
    
    val = np.zeros((T+1, capacity+1))
    result = np.zeros((capacity+1))
    for cap in range(1, capacity+1):
        result[cap] = evaluate_msecretary(val, sol_index, prob_choice, rewards, T, cap)
    
    return result[capacity], val


if __name__ == '__main__':
    np.random.seed(42)
    n_types = 4
    capacity = 99 #capacity. benchmark = 5
    T = 100 #Time periods remaining. benchmark = 10

    probabilities = uniform.rvs(size = n_types)
    probabilities /= probabilities.sum()
    rewards = uniform.rvs(scale = 10, size = n_types)

    vectors = generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = dynamic_solution(T, capacity)
    #print(result_dynamic, val_dynamic)

    result_approx, val_approx, sol_approx, sol_index_approx = approx_dynamic_solution(T, capacity)
    #print(result_approx, val_approx)

    result_eval_approx, val_eval_approx = evaluate_solution(T, capacity, sol_index_approx)
    #print(result_eval_approx, val_eval_approx)

    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/'

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_approx[fix_t], color = 'tab:red', label='Value function using bellman approximation')
        plt.plot(val_eval_approx[fix_t], color = 'tab:blue', linestyle= '-', marker = '.', label = 'Value function using solutions from bellman approximation')

        # Decoration
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.title('Value function "$V_t(x)$" of multi-secretary problem with ' + str(n_types) +' types for t = '+str(T-fix_t), fontsize=20)
        plt.grid(axis='both', alpha=.3)
        plt.xlabel('x (capacity)', fontsize = 14)
        
        # Remove borders
        plt.gca().spines["top"].set_alpha(0.3)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.3)    
        plt.gca().spines["left"].set_alpha(0.3)   
        
        plt.legend(loc = "lower right")
        
        plt.savefig(path+'t_period'+str(T-fix_t)+'.png')
        plt.clf()

