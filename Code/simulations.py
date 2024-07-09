import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import uniform
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os 
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

    #probabilities = uniform.rvs(size = n_types)
    #probabilities /= probabilities.sum()
    probabilities = np.array([.25, .25, .25, .25]) 
    #probabilities = np.array([.125, .125, .125, .125, .125, .125, .125, .125]) 
    #rewards = np.array([4, 2, .5, 9])
    rewards = np.array([.5, 1, 1.5, 2])
    #rewards = np.array([.2, .4, .6, .8, 1, 1.2, 1.4, 1.6])

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    ########

    suboptimality_gap = {}
    max_suboptimality_gap = {}
    max_suboptimality_gap_t = {}
    which_t_max = {}
    which_x_max = {}
    result_lookahead = {}
    val_lookahead = {}
    sol_lookahead = {}
    sol_index_lookahead = {}
    result_eval_lookahead = {}
    val_eval_lookahead = {}

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_solution(T, capacity, prob_choice, rewards, vectors)
    val_deterministic = ms.deterministic_msecretary_array(T, capacity, np.arange(1, T+1), probabilities, rewards, n_types)
    result_lookahead[1], val_lookahead[1], sol_lookahead[1], sol_index_lookahead[1] = ms.approx_dynamic_solution(T, capacity, val_deterministic, prob_choice, rewards, vectors)
    result_eval_lookahead[1], val_eval_lookahead[1] = ms.evaluate_solution(T, capacity, sol_index_lookahead[1], prob_choice, rewards)
    windows = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    suboptimality_gap[1] = np.divide(val_dynamic-val_eval_lookahead[1], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
    max_suboptimality_gap[1] = np.max(suboptimality_gap[1][T])
    max_suboptimality_gap_t[1] = np.max(suboptimality_gap[1])
    which_t_max[1], which_x_max[1] = np.unravel_index(np.argmax(suboptimality_gap[1]), suboptimality_gap[1].shape)
    
    for window in tqdm(windows):
        result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
        result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
        suboptimality_gap[window] = np.divide(val_dynamic-val_eval_lookahead[window], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
        max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
        max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
        which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)
    
    windows_plot = [1, 10, 50]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/suboptimality_gap/percentage'#+str(probabilities)+str(rewards)
    
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(16,10), dpi= 80)
    for dix, win in enumerate(windows_plot):
        plt.plot(suboptimality_gap[win][T], linestyle= '-', label = 'lookahead = '+str(win))
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Suboptimality Gap Lookahead Approximation', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Remaining capacity', fontsize = 14)
    plt.legend(loc = "lower right")
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)       
    
    plt.savefig(path+'/remaining_capacity.pdf')
    plt.clf()
    ############################################################################################################
    plt.figure(figsize=(16,10), dpi= 80)
    # Sort the dictionary by keys
    sorted_data = dict(sorted(max_suboptimality_gap.items()))
    # Extract keys and values
    x = list(sorted_data.keys())
    y = list(sorted_data.values())
    # Create the line plot
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
             markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
     # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Lookahead Heuristic Maximum Suboptimality Gap', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Number of lookahead steps', fontsize = 14)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)       
    
    plt.savefig(path+'/maximum_sub_gap.pdf')
    plt.clf()


    ############################################################################################################

    periods_plot = [0, 25, 50, 75]
    windows_plot = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    for t in periods_plot:
        for dix, win in enumerate(windows_plot):   
            df = pd.DataFrame(sol_lookahead[win][100-t], columns=['$r_4=.5$', '$r_3=1$', '$r_2=1.5$', '$r_1=2$'])
            new_labels = [i / (100-t) for i in range(df.shape[0])]

            # Plotting the heatmap
            plt.figure(figsize=(16,10), dpi= 80)
            sns.heatmap(df[100-t:].T, cmap='bwr', cbar=False, annot=False, linewidths=0.5)
            plt.xlabel('Ratio capacity/remaining periods')
            plt.ylabel('Reward type')
            plt.title('Action Map on T='+str(t)+' (Initial Period) for n-lookahead='+str(win))
            plt.xticks(ticks=np.arange(df.shape[0]) + 0.5, labels=new_labels[100-t], rotation=90)
            path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/action_map/'+'period_'+str(t)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+'/action_map_'+str(win)+'.png')
