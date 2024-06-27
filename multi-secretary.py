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

    result = msecretary(val, sol_index, prob_choice, rewards, T, capacity)
    sol = vectors[sol_index]

    return result, val, sol 

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

    return result, val, sol


if __name__ == '__main__':
    np.random.seed(42)
    n_types = 4
    capacity = 5 #capacity. benchmark = 5
    T = 10 #Time periods remaining. benchmark = 10

    probabilities = uniform.rvs(size = n_types)
    probabilities /= probabilities.sum()
    rewards = uniform.rvs(scale = 10, size = n_types)

    vectors = generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.

    result_dynamic, val_dynamic, sol_dynamic = dynamic_solution(T, capacity)
    print(result_dynamic, val_dynamic, sol_dynamic)

    result_approx, val_approx, sol_approx = approx_dynamic_solution(T, capacity)
    print(result_approx, val_approx, sol_approx)

    #print(deterministic_msecretary(probabilities, rewards, n_types, T, capacity))
    


