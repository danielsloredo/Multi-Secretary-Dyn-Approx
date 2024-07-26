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

if __name__ == '__main__':
    ######## This are global variables
    n_types = 4
    #capacity = 100 #capacity upper bound
    #T = 100 #Total time of observation
    probabilities = np.array([.25, .25, .25, .25])#np.array([.5, .5]) 
    rewards = np.array([.5, 1, 1.5, 2])#np.array([ 1, 2])#

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    window = 2
    n_sims = 5000
    horizons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
    ########
    regret = {}
    regret_opt = {}
    val_offline = {}
    
    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/test_2/regret_2_lookahead'
    
    if not os.path.exists(path):
        os.makedirs(path)

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_msecretary(horizons[-1], horizons[-1], probabilities, rewards, vectors)
    val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)
    

    for horizon in tqdm(horizons):
        capacity = horizon
        result_lookahead, val_lookahead, sol_lookahead, sol_index_lookahead = ms.dynamic_msecretary_lookahead(horizon, capacity, val_deterministic, window, probabilities, rewards, vectors)
        result_eval_lookahead, val_eval_lookahead = ms.dynamic_evaluate_solution(horizon, capacity, sol_lookahead, probabilities, rewards)
        values_offline = ms.simulate_offline_msecretary(horizon, horizon, probabilities, rewards, n_types, n_sims, seed = 42)
        val_offline[horizon] = values_offline.mean(axis = 0)
        regret[horizon] = np.max(val_offline[horizon]-val_eval_lookahead[horizon])
        regret_opt[horizon] = np.max(val_offline[horizon]-val_dynamic[horizon][:horizon+1])
        
    
        
    plt.figure(figsize=(16,10), dpi= 80)
    # Extract keys and values
    x = list(regret.keys())
    y = list(regret.values())
    z = list(regret_opt.values())
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
            markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Lookahead Policy')
    plt.plot(x, z, color='tab:blue', marker='+', markersize=5, 
            markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:blue', label = 'Optimal Policy')
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Regret For Restricted Lookahead Policy on t=0', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Horizon', fontsize = 14)
    plt.legend()
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)

    plt.savefig(path+'/regret_absolute_increasing_horizon.png')
    plt.close()

    
