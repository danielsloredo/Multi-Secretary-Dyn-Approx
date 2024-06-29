import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import uniform
import sys
# Add the directory to sys.path
module_path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Code/'
sys.path.append(module_path)
# Now you can import your module
import multi_secretary as ms

if __name__ == '__main__':
    np.random.seed(42)
    ######## This are global variables
    n_types = 4
    capacity = 99 #capacity. benchmark = 5
    T = 100 #Time periods remaining. benchmark = 10
    window = 5

    probabilities = uniform.rvs(size = n_types)
    probabilities /= probabilities.sum()
    rewards = np.array([4, 2, .5, 9]) #uniform.rvs(scale = 10, size = n_types)

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    ########

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_solution(T, capacity, prob_choice, rewards, vectors)
    #print(result_dynamic, val_dynamic)

    val_deterministic = ms.deterministic_msecretary_array(T, capacity, np.arange(1, T+1), probabilities, rewards, n_types)

    result_approx, val_approx, sol_approx, sol_index_approx = ms.approx_dynamic_solution(T, capacity, val_deterministic, prob_choice, rewards, vectors)
    #print(result_approx, val_approx)

    result_eval_approx, val_eval_approx = ms.evaluate_solution(T, capacity, sol_index_approx, prob_choice, rewards)
    #print(result_eval_approx, val_eval_approx)

    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/1_step/'

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

    
    result_lookahead, val_lookahead, sol_lookahead, sol_index_lookahead = ms.approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
    #print(result_approx, val_approx) 

    result_eval_lookahead, val_eval_lookahead = ms.evaluate_solution(T, capacity, sol_index_lookahead, prob_choice, rewards)
    #print(result_eval_approx, val_eval_approx)

    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/'+str(window)+'_step/'

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_lookahead[fix_t], color = 'tab:red', label='Value function using bellman approximation')
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
        
        plt.savefig(path+'t_period'+str(T-fix_t)+'.png')
        plt.clf()
