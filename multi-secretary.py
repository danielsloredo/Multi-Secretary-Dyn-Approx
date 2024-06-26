import numpy as np 
import matplotlib.pyplot as plt
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

def msecretary(prob_choice, rewards, t, B): 
    if t == 0 or B == 0:
        return 0 
    if val[t][B] != 0: 
        return val[t][B]
    
    val[t][B] = max(np.sum(prob_choice * (rewards + msecretary(prob_choice, rewards, t-1, B-1)), axis = 1)
                    + (1- prob_choice.sum(axis =1))*msecretary(prob_choice,rewards, t-1, B))
    
    return val[t][B]

if __name__ == '__main__':
    np.random.seed(42)
    n_types = 3
    probabilities = uniform.rvs(size = n_types)
    probabilities /= probabilities.sum()
    rewards = uniform.rvs(scale = 10, size = n_types)

    capacity = 10 #capacity. benchmark = 5
    T = 20 #Time periods. benchmark = 10

    vectors = generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables. 

    val = np.zeros((T+1, capacity+1))

    print(msecretary(prob_choice, rewards, T, capacity))




