import numpy as np 
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os 
import sys
module_path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Code/'
sys.path.append(module_path)
import multi_secretary as ms
sys.setrecursionlimit(3500)


######## This are global variables
n_types = 4
#capacity = 100 #capacity upper bound
#T = 100 #Total time of observation
probabilities = np.array([.25, .25, .25, .25])#np.array([.5, .5]) 
rewards = np.array([.5, 1, 1.5, 2])# np.array([1, 2]) #

vectors = ms.generate_vectors(n_types)
prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.

n_sims = 5000
horizons = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000]#, 1200, 1400, 1600, 2000, 3000]
window = {}
window = {i: int(np.floor(i**(2/3))) for i in horizons}
########
regret = {}
regret_opt = {}
val_offline = {}
sub_opt_gap = {}


result_lookahead = {}
val_lookahead = {}
sol_lookahead = {}
sol_index_lookahead = {}
result_eval_lookahead = {}
val_eval_lookahead = {}

path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'
path_data = path_0 + 'Data/4_types/regret_2_3/'
path_offline = path_0 + 'Data/4_types/'
path = path_0 + 'Figures/4_types/regret_2_3_lookahead/'
path_1 = path_0 + 'Figures/4_types/value_functions/2_3_lookahead/'

#result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_msecretary(horizons[-1], horizons[-1], probabilities, rewards, vectors)
#val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)
#values_offline = ms.simulate_offline_msecretary(horizons[-1], horizons, probabilities, rewards, n_types, n_sims, seed = 42)
#for horizon in horizons:
#    val_offline[horizon] = values_offline[horizon].mean(axis = 0)

with open(path_data+'val_lookahead.pkl', 'rb') as pickle_file:
    val_lookahead = pickle.load(pickle_file)
with open(path_data+'sol_lookahead.pkl', 'rb') as pickle_file:
    sol_lookahead = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead.pkl', 'rb') as pickle_file:
    sol_index_lookahead = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead.pkl', 'rb') as pickle_file:
    val_eval_lookahead = pickle.load(pickle_file)
with open(path_offline+'val_offline.pkl', 'rb') as pickle_file:
    val_offline = pickle.load(pickle_file)

'''
if not os.path.exists(path_data):
    os.makedirs(path_data)
with open(path_data+'val_offline.pkl', 'wb') as pickle_file:
    pickle.dump(val_offline, pickle_file)


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
    with open(path_data+'val_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(val_lookahead, pickle_file)
    with open(path_data+'sol_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(sol_lookahead, pickle_file)
    with open(path_data+'sol_index_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(sol_index_lookahead, pickle_file)
    with open(path_data+'val_eval_lookahead.pkl', 'wb') as pickle_file:
        pickle.dump(val_eval_lookahead, pickle_file)
'''    

for horizon in horizons:
    regret[horizon] = np.max(val_offline[horizon]-val_eval_lookahead[horizon][horizon][:horizon+1])
    regret_opt[horizon] = np.max(val_offline[horizon]-val_dynamic[horizon][:horizon+1])
    sub_opt_gap[horizon] = np.max(val_dynamic[horizon][:horizon+1]-val_eval_lookahead[horizon][horizon][:horizon+1])
    
    
if not os.path.exists(path):
    os.makedirs(path)

plt.figure(figsize=(16,10), dpi= 80)
# Extract keys and values
sorted_data = dict(sorted(sub_opt_gap.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'DP-Lookahead')
# Decoration
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Suboptimality Gap for Restricted $n^{3/5}$ Lookahead Policy on t=0', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
# Remove borders
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)

plt.savefig(path+'subopt_increasing_horizon.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
# Extract keys and values
sorted_data = dict(sorted(regret.items()))
#sorted_data2 = dict(sorted(regret_opt.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
#z = list(sorted_data2.values())
plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Lookahead Policy')
#plt.plot(x, z, color='tab:blue', marker='+', markersize=5, 
#        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:blue', label = 'Optimal Policy')
# Decoration
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Regret for Restricted $n^{3/5}$ Lookahead Policy on t=0', fontsize=20)
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


if not os.path.exists(path_1):
    os.makedirs(path_1)



t_periods = [i for i in np.arange(100, horizons[-1]+1, 100)]

for dix, fix_t in enumerate(horizons):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_dynamic[fix_t][:fix_t+1], color = 'black', label='Optimal Value Function', linestyle = '-')
    plt.plot(val_deterministic[fix_t][:fix_t+1], color = 'tab:red', label='LP Upper Bound')
    plt.plot(val_lookahead[fix_t][fix_t][:fix_t+1], color = 'tab:blue', label='Lookahead DP', linestyle = 'dashdot', marker = '', fillstyle = 'none')
    plt.plot(val_eval_lookahead[fix_t][fix_t][:fix_t+1], color = 'tab:green', label='Lookahead Heuristic', linestyle = 'dotted', marker = '', fillstyle = 'none')
    
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Value function "$V_t(x)$" of multi-secretary problem with ' + str(n_types) +' types for remaining periods t = '+str(fix_t), fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('x (capacity)', fontsize = 14)
    
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.savefig(path_1+'/value_functions_'+str(fix_t)+'.png')
    plt.close()
    
for dix, fix_t in enumerate(horizons):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_deterministic[fix_t][:fix_t+1]-val_dynamic[fix_t][:fix_t+1], color = 'tab:red', label='Difference LP-DP', marker='+')
    plt.plot(val_lookahead[fix_t][fix_t][:fix_t+1]-val_dynamic[fix_t][:fix_t+1], color = 'tab:blue', label='Difference Lookahead DP-DP', marker='o', fillstyle = 'none')
    plt.plot(val_dynamic[fix_t][:fix_t+1]-val_eval_lookahead[fix_t][fix_t][:fix_t+1], color = 'tab:gray', label='Difference DP-Heuristic', marker=',', fillstyle = 'none')
    
    # Decoration
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title('Difference in Value function "$V_t(x)$" of multi-secretary problem with ' + str(n_types) +' types for remaining periods t = '+str(fix_t), fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('x (capacity)', fontsize = 14)
    
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.3)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.3)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend(loc = "lower right")
    plt.savefig(path_1+'/diff/diff_value_functions_'+str(fix_t)+'.png')
    plt.close()
