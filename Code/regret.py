import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os 
import sys
module_path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Code/'
sys.path.append(module_path)
import multi_secretary as ms
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    ######## This are global variables
    n_types = 2
    #capacity = 100 #capacity upper bound
    #T = 100 #Total time of observation
    probabilities = np.array([.5, .5])#np.array([.25, .25, .25, .25]) 
    rewards = np.array([ 1, 2])#np.array([.5, 1, 1.5, 2])

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    window = 1
    horizons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
    ########
    regret = {}
    regret_percentage = {}

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/regret_1_lookahead'
    
    if not os.path.exists(path):
        os.makedirs(path)

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_solution(horizons[-1], horizons[-1], probabilities, rewards, vectors)
    val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)

    for horizon in tqdm(horizons):
        capacity = horizon
        result_lookahead, val_lookahead, sol_lookahead, sol_index_lookahead = ms.approx_n_lookahead(horizon, capacity, val_deterministic, window, probabilities, rewards, vectors)
        result_eval_lookahead, val_eval_lookahead = ms.evaluate_solution(horizon, capacity, sol_index_lookahead, prob_choice, rewards)
        regret[horizon] = np.max(val_dynamic[:horizon+1, :horizon+1]-val_eval_lookahead, axis = 1)
        #regret_percentage[horizon] = np.divide(regret[horizon], 
        #                                   val_dynamic[:horizon+1, :horizon+1],
        #                                   out = np.zeros_like(val_dynamic[:horizon+1, :horizon+1]),
        #                                   where= val_dynamic[:horizon+1, :horizon+1]!=0)
        

    regret_initial = {}
    regret_initial_percentage = {}
    for horizon in horizons:
        regret_initial[horizon] = regret[horizon][-1] 
        #regret_initial_percentage[horizon] = regret_percentage[horizon][-1]

    plt.figure(figsize=(16,10), dpi= 80)
    # Extract keys and values
    x = list(regret_initial.keys())
    y = list(regret_initial.values())
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
            markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Regret For Restricted Lookahead Policy on t=0', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Horizon', fontsize = 14)
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)

    plt.savefig(path+'/regret_absolute_increasing_horizon.png')
    plt.close()

    #plt.figure(figsize=(16,10), dpi= 80)
    # Extract keys and values
    #x = list(regret_initial_percentage.keys())
    #y = list(regret_initial_percentage.values())
    #plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
    #        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
    # Decoration
    #plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    #plt.yticks(fontsize=12, alpha=.7)
    #plt.title('Regret (Percentage) For Restricted Lookahead Policy on t=0', fontsize=20)
    #plt.grid(axis='both', alpha=.3)
    #plt.xlabel('Horizon', fontsize = 14)
    # Remove borders
    #plt.gca().spines["top"].set_alpha(0.3)    
    #plt.gca().spines["bottom"].set_alpha(0.3)
    #plt.gca().spines["right"].set_alpha(0.3)    
    #plt.gca().spines["left"].set_alpha(0.3)

    #plt.savefig(path+'/regret_increasing_horizon.png')
    #plt.close()
