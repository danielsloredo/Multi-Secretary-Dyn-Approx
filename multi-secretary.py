import numpy as np 
from scipy.stats import uniform

###############################################################
# Dynamic programing approach to multisecreatry problem with n 
# types, only one type i apearing each time step with prob p_i
###############################################################
np.random.seed(42)

n_types = 3
probabilities = uniform.rvs(size = n_types)
probabilities /= probabilities.sum()
rewards = uniform.rvs(scale = 10, size = n_types)

capacity = 5 #capacity
T = 10 #Time periods

vectors = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables. 

val = np.zeros((T+1, capacity+1))


def msecretary(prob_choice, rewards, t, B): 
    if t == 0 or B == 0:
        return 0 
    if val[t][B] != 0: 
        return val[t][B]
    
    val[t][B] = max(np.sum(prob_choice * (rewards + msecretary(prob_choice, rewards, t-1, B-1)), axis = 1)
                    + (1- prob_choice.sum(axis =1))*msecretary(prob_choice,rewards, t-1, B))
    
    return val[t][B]

print(msecretary(prob_choice, rewards, T, capacity))






