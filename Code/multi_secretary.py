import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################
# Dynamic programing approach to multisecreatry problem with n 
# types, only one type i apearing each time step with prob p_i
###############################################################
def generate_vectors(n):
    vectors = []
    for i in range(n + 1):
        vector = [0] * (n - i) + [1] * i
        vectors.append(vector)
    
    return np.array(vectors)

def deterministic_msecretary(probabilities, rewards, n_types, t, x):
    #Linear programming relaxation of the multi-secretary problem. (The deterministic version)
    y_temp = np.zeros(n_types)
    restriction = np.multiply(probabilities[::-1], t)
    cumsum = np.cumsum(restriction)
    logic_test = cumsum <= x
    if logic_test.sum() == 0:
        y_temp[0] = x
    else:
        i_max = np.sum(logic_test) - 1
        y_temp[:i_max+1] = restriction[:i_max+1]
        if i_max + 1 < n_types:
            y_temp[i_max+1] = x - np.sum(y_temp[:i_max+1]) 
    
    y = y_temp[::-1]
    result = np.sum(np.multiply(rewards, y))

    return result

def deterministic_msecretary_array(T, capacity, probabilities, rewards, n_types):
    #Solve the deterministic version of the multi-secretary problem for all periods in approx_periods and all capacities
    val_deterministic = np.zeros((T+1, capacity+1))

    for t in range(1, T+1):
        for x in range(1, capacity+1):
            val_deterministic[t][x] = deterministic_msecretary(probabilities, rewards, n_types, t, x)

    return val_deterministic

def msecretary(val, sol, probabilities, rewards, flag_computed, t, x): 
    #Bellman recursion for the multi-secretary problem
    if t == 0 or x == 0:
        return 0 
    if flag_computed[t][x] != 0: 
        return val[t][x]
    
    next_less = msecretary(val, sol, probabilities, rewards, flag_computed, t-1, x-1)
    next_same = msecretary(val, sol, probabilities, rewards, flag_computed, t-1, x)

    logic_test = (rewards + next_less >= next_same)
    logic_test_2 = (rewards + next_less > next_same)
    if logic_test_2.sum() == 0 and logic_test.sum() > 0:
        #I only want the highest one
        logic_test_3 = np.full(rewards.shape[0], False)
        logic_test_3[np.argmax(rewards)] = True
        q_val = np.where(logic_test_3, rewards + next_less, next_same)
        sol[t][x] = np.where(logic_test_3, 1, 0)  
    else:
        q_val = np.where(logic_test_2, rewards + next_less, next_same)
        sol[t][x] = np.where(logic_test_2, 1, 0)
        
    val[t][x] = np.sum(np.multiply(probabilities, q_val))
    flag_computed[t][x] = 1
    
    return val[t][x]

def msecretary_lookahead(val, sol, probabilities, rewards, flag_computed, t, x, val_deterministic, window, depth = 0): 
    #Bellman recursion for the multi-secretary problem with n-lookahead
    if t == 0 or x == 0:
        return 0 
    if flag_computed[t][x] != 0: 
        return val[t][x]
    if depth == window: 
        return val_deterministic[t][x]
    
    next_less = msecretary_lookahead(val, sol, probabilities, rewards, flag_computed, t-1, x-1, val_deterministic, window, depth+1)
    next_same = msecretary_lookahead(val, sol, probabilities, rewards, flag_computed, t-1, x, val_deterministic, window, depth+1)

    #logic_test = (rewards + next_less >= next_same)
    logic_test_2 = (rewards + next_less > next_same)
    #if logic_test_2.sum() == 0:
    #    #I only want the highest one
    #    logic_test_3 = np.full(rewards.shape[0], False)
    #    logic_test_3[np.argmax(rewards)] = True
    #    q_val = np.where(logic_test_3, rewards + next_less, next_same)
    #    sol[t][x] = np.where(logic_test_3, 1, 0)  
    #else: 
    
    q_val = np.where(logic_test_2, rewards + next_less, next_same)
    sol[t][x] = np.where(logic_test_2, 1, 0)

    val[t][x] = np.sum(np.multiply(probabilities, q_val))
    flag_computed[t][x] = 1
        
    return val[t][x]

def dynamic_solution(T, capacity, probabilities, rewards, vectors):
    #Dynamic programing solution to the multi-secretary problem
    val = np.zeros((T+1, capacity+1))
    flag_computed = np.zeros((T+1, capacity+1))
    sol = np.zeros((T+1, capacity+1, rewards.shape[0]), dtype=int)
    result = np.zeros((capacity+1))

    for cap in reversed(np.arange(1, capacity+1)):
        result[cap] = msecretary(val, sol, probabilities, rewards, flag_computed, T, cap)
    
    comparison = sol[..., np.newaxis, :] == vectors
    matches = np.all(comparison, axis=-1)
    sol_index = np.argmax(matches, axis=-1)
    
    return result, val, sol, sol_index

def approx_n_lookahead(T, capacity, val_deterministic, window, probabilities, rewards, vectors):
    #Approximate dynamic programing solution to the multi-secretary problem with n-lookahead
    value = np.zeros((T+1, capacity+1))
    result = np.zeros((T+1, capacity+1))
    solution = np.zeros((T+1, capacity+1, rewards.shape[0]), dtype=int)

    for period in range(1, T+1): 
        val_temp = np.zeros((T+1, capacity+1))
        sol_temp = np.zeros((T+1, capacity+1, rewards.shape[0]), dtype=int)
        flag_computed_temp = np.zeros((T+1, capacity+1))

        for cap in range(1, capacity+1):
            result[period][cap] = msecretary_lookahead(val_temp, sol_temp, probabilities, rewards, flag_computed_temp, period, cap, val_deterministic, window)
            value[period][cap] = val_temp[period][cap]
            solution[period][cap] = sol_temp[period][cap]
    
    comparison = solution[..., np.newaxis, :] == vectors
    matches = np.all(comparison, axis=-1)
    sol_index = np.argmax(matches, axis=-1)

    return result, value, solution, sol_index

def approx_dynamic_solution(T, capacity, val_deterministic, prob_choice, rewards, vectors):
    #Approximate dynamic programing solution to the multi-secretary problem (1-lookahead)

    val = np.zeros((T+1, capacity+1))
    sol_index = np.zeros((T+1, capacity+1), dtype=int)
    
    for t in range(1, T+1):
        for x in range(1, capacity+1):
            if t == 0 or x == 0:
                val[t][x] = 0 
    
            q_val = (np.sum(prob_choice * (rewards + val_deterministic[t-1][x-1]), axis = 1) 
                     + (1-prob_choice.sum(axis = 1))*val_deterministic[t-1][x])
            sol_index[t][x] = np.argmax(q_val)
            val[t][x] = q_val[sol_index[t][x]]

    result = val[T][capacity]
    sol = vectors[sol_index]

    return result, val, sol, sol_index

def evaluate_msecretary(val, sol_index, prob_choice, rewards, t, x): 
    #Evaluate approximate solution on the original bellman recursion for the multi-secretary 
    if t == 0 or x == 0:
        return 0 
    if val[t][x] != 0: 
        return val[t][x]
    
    prob_chosen = prob_choice[sol_index[t][x]]
    q_val = (np.sum(prob_chosen * (rewards + evaluate_msecretary(val, sol_index, prob_choice, rewards, t-1, x-1))) 
             + (1-prob_chosen.sum())*evaluate_msecretary(val, sol_index, prob_choice, rewards, t-1, x))
    
    val[t][x] = q_val
    
    return val[t][x]

def evaluate_solution(T, capacity, sol_index, prob_choice, rewards):
    #Obtain the value funtion of the approximation solution
    val = np.zeros((T+1, capacity+1))
    result = np.zeros((capacity+1))
    for cap in range(1, capacity+1):
        result[cap] = evaluate_msecretary(val, sol_index, prob_choice, rewards, T, cap)
    
    return result[capacity], val


if __name__ == '__main__':
    
    ######## This are global variables
    n_types = 4
    capacity = 99 #capacity. benchmark = 5
    T = 100 #Time periods remaining. benchmark = 10

    probabilities = np.array([.25, .25, .25, .25])
    rewards = np.array([.5, 2, 4, 9]) #uniform.rvs(scale = 10, size = n_types)
    vectors = generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    ########

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = dynamic_solution(T, capacity, prob_choice, rewards,vectors)
    val_deterministic = deterministic_msecretary_array(T, capacity, probabilities, rewards, n_types)
    result_approx, val_approx, sol_approx, sol_index_approx = approx_dynamic_solution(T, capacity, val_deterministic, prob_choice, rewards, vectors)
    result_eval_approx, val_eval_approx = evaluate_solution(T, capacity, sol_index_approx, prob_choice, rewards)
    #print(result_eval_approx, val_eval_approx)

    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_deterministic[fix_t], color = 'tab:red', label='Value function using bellman approximation')
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
        
        plt.show()

    
    window = 5
    result_lookahead, val_lookahead, sol_lookahead, sol_index_lookahead = approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
    result_eval_lookahead, val_eval_lookahead = evaluate_solution(T, capacity, sol_index_lookahead, prob_choice, rewards)
    
    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_deterministic[fix_t], color = 'tab:red', label='Value function using bellman approximation')
        plt.plot(val_eval_lookahead[fix_t], color = 'tab:blue', linestyle= '-', marker = '.', label = 'Value function using solutions from bellman approximation')

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
        
        plt.show()

