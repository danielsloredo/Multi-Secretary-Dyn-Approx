import numpy as np 
import matplotlib.pyplot as plt
import pickle
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
    
    n_sims = 5000
    horizons = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000]
    window = {}
    window = {i: int(np.floor(i**(11/20))) for i in horizons}
    ########
    regret = {}
    regret_opt = {}
    val_offline = {}

    
    result_lookahead = {}
    val_lookahead = {}
    sol_lookahead = {}
    sol_index_lookahead = {}
    result_eval_lookahead = {}
    val_eval_lookahead = {}
    
    path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'

    result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_msecretary(horizons[-1], horizons[-1], probabilities, rewards, vectors)
    val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)
    values_offline = ms.simulate_offline_msecretary(horizons[-1], horizons, probabilities, rewards, n_types, n_sims, seed = 42)
    

    for horizon in tqdm(horizons):
        capacity = horizon
        (result_lookahead[horizon], 
         val_lookahead[horizon], 
         sol_lookahead[horizon], 
         sol_index_lookahead[horizon]) = ms.dynamic_msecretary_lookahead(
             horizon, 
             capacity, 
             val_deterministic, 
             window[horizon], 
             probabilities, 
             rewards, 
             vectors)
        (result_eval_lookahead[horizon], 
         val_eval_lookahead[horizon]) = ms.dynamic_evaluate_solution(
             horizon, 
             capacity, 
             sol_lookahead[horizon], 
             probabilities, 
             rewards)
        
    path_data = path_0 + 'Data/4_types/regret_11_20/'
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    with open(path_data+'val_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(val_lookahead, pickle_file)
    with open(path_data+'sol_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(sol_lookahead, pickle_file)
    with open(path_data+'sol_index_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(sol_index_lookahead, pickle_file)
    with open(path_data+'val_eval_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(val_eval_lookahead, pickle_file)
    
    '''
    with open(path_data+'val_lookahead.pkl', 'rb') as pickle_file:
        val_lookahead = pickle.load(pickle_file)
    with open(path_data+'sol_lookahead.pkl', 'rb') as pickle_file:
        sol_lookahead = pickle.load(pickle_file)
    with open(path_data+'sol_index_lookahead.pkl', 'rb') as pickle_file:
        sol_index_lookahead = pickle.load(pickle_file)
    with open(path_data+'val_eval_lookahead.pkl', 'rb') as pickle_file:
        val_eval_lookahead = pickle.load(pickle_file)
    '''

    for horizon in horizons:
        val_offline[horizon] = values_offline[horizon].mean(axis = 0)
        regret[horizon] = np.max(val_offline[horizon]-val_eval_lookahead[horizon][horizon])
        regret_opt[horizon] = np.max(val_offline[horizon]-val_dynamic[horizon][:horizon+1])
        
    
    path = path_0 + 'Figures/4_types/regret_11_20_lookahead/' 
    if not os.path.exists(path):
        os.makedirs(path)
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
    plt.title('Regret For Restricted $n^{11/20}$ Lookahead Policy on t=0', fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Horizon', fontsize = 14)
    plt.legend()
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)

    plt.savefig(path+'regret_increasing_horizon.png')
    plt.close()

    