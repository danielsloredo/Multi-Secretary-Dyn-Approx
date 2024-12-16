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



######## This are global variables
n_types = 4
capacity = 100 #capacity upper bound
T = 100 #Total time of observation
n_sims = 5000
probabilities = np.array([.25, .25, .25, .25]) 
rewards = np.array([.5, 1, 1.5, 2]) #* (10**10)

vectors = ms.generate_vectors(n_types)
prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
windows = [i for i in np.arange(5, T+5, 5)]
windows.insert(0, 1)
path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/'
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

result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_solution(T, capacity, probabilities, rewards, vectors)
val_deterministic = ms.deterministic_msecretary_array(T, capacity, probabilities, rewards, n_types)
values_offline = ms.simulate_offline_msecretary(T, [capacity], probabilities, rewards, n_types, n_sims, seed = 42)
val_offline = values_offline[100].mean(axis = 0)

for window in tqdm(windows):
    result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, probabilities, rewards, vectors)
    result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
    suboptimality_gap[window] = val_dynamic-val_eval_lookahead[window]#, val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
    max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
    max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
    which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#Value functions
path = path_0 + 'value_functions/LP_Optimal'
if not os.path.exists(path):
    os.makedirs(path)

t_periods = [i for i in np.arange(T, T+5, 5)]

for dix, fix_t in enumerate(t_periods):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_dynamic[fix_t], color = 'tab:grey', label='Optimal value function', linestyle = '-')
    plt.plot(val_deterministic[fix_t], color = 'tab:red', label='LP Upper Bound')
    plt.plot(val_offline, color = 'tab:blue', label='Offline value function', linestyle = 'dashdot', marker = '', fillstyle = 'none')
    
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
    plt.savefig(path+'/value_functions_'+str(fix_t)+'.png')
    plt.savefig(path+'/value_functions_'+str(fix_t)+'.pdf')
    plt.close()

for dix, fix_t in enumerate(t_periods):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_deterministic[fix_t]-val_dynamic[fix_t], color = 'tab:red', label='Difference LP-DP', marker='+')
    plt.plot(val_offline-val_dynamic[fix_t], color = 'tab:blue', label='Difference Offline-DP', marker='o', fillstyle = 'none')
    
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
    plt.savefig(path+'/diff_value_functions_'+str(fix_t)+'.png')
    plt.savefig(path+'/diff_value_functions_'+str(fix_t)+'.pdf')
    plt.close()


path = path_0 + 'value_functions/25_lookahead'
step = 25
if not os.path.exists(path):
    os.makedirs(path)

t_periods = [i for i in np.arange(T, T+5, 5)]

for dix, fix_t in enumerate(t_periods):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal Value Function', linestyle = '-')
    plt.plot(val_deterministic[fix_t], color = 'tab:red', label='LP Upper Bound')
    plt.plot(val_lookahead[step][fix_t], color = 'tab:blue', label='Lookahead DP', linestyle = 'dashdot', marker = '', fillstyle = 'none')
    plt.plot(val_eval_lookahead[step][fix_t], color = 'tab:green', label='Lookahead Heuristic', linestyle = 'dotted', marker = '', fillstyle = 'none')
    plt.plot(val_offline, color = 'tab:orange', label='Offline value function', linestyle = '--', marker = '', fillstyle = 'none')
    
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
    plt.savefig(path+'/value_functions_'+str(fix_t)+'.png')
    plt.close()
    
for dix, fix_t in enumerate(t_periods):
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(val_deterministic[fix_t]-val_dynamic[fix_t], color = 'tab:red', label='Difference LP-DP', marker='+')
    plt.plot(val_lookahead[step][fix_t]-val_dynamic[fix_t], color = 'tab:blue', label='Difference Lookahead DP-DP', marker='o', fillstyle = 'none')
    plt.plot(val_offline-val_eval_lookahead[step][fix_t], color = 'tab:orange', label='Difference Offline-Heuristic', marker='x', fillstyle = 'none')
    plt.plot(val_offline-val_dynamic[fix_t], color = 'black', label='Difference Offline-DP', marker='', fillstyle = 'none')
    plt.plot(val_dynamic[fix_t]-val_eval_lookahead[step][fix_t], color = 'tab:gray', label='Difference DP-Heuristic', marker=',', fillstyle = 'none')
    
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
    plt.savefig(path+'/diff/diff_value_functions_'+str(fix_t)+'.png')
    plt.close()

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#Subotimality gap plots 
windows_plot = [1, 10, 50]

path = path_0 + 'suboptimality_gap'

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
plt.close()

#########################################################################################################################

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
plt.close()


############################################################################################################
############################################################################################################
############################################################################################################
#Action maps
cmap_dict = {0: 'tab:blue', 1: 'lavender', 2: 'tab:orange', 3: 'tab:grey', 4: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(5)])
df = pd.DataFrame(sol_index_dynamic)
#df_reversed_rows = df.iloc[::-1, :]
df_reversed_cols = df.iloc[:, ::-1]
    # Plotting the heatmap
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed_cols, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
plt.xlabel('Remaining Capacity')
plt.ylabel('Remaining Time')
plt.title('Action Map for Optimal Solution')

patch_0 = mpatches.Patch(color='tab:blue', label='None')
patch_1 = mpatches.Patch(color='lavender', label='Highest Type')
patch_2 = mpatches.Patch(color='tab:orange', label='2 Highest Types')
patch_3 = mpatches.Patch(color='tab:grey', label='3 Highest Types')
patch_4 = mpatches.Patch(color='tab:red', label='All Types')
plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], loc='upper right', bbox_to_anchor=(1.12, 1))

path = path_0 + 'action_map/optimal'
if not os.path.exists(path):
    os.makedirs(path)
plt.savefig(path+'/action_map_.png')
plt.close()

for dix, win in enumerate(windows):   
    df = pd.DataFrame(sol_index_lookahead[win])
    #df_reversed_rows = df.iloc[::-1, :]
    df_reversed_cols = df.iloc[:, ::-1]
    # Plotting the heatmap
    plt.figure(figsize=(16,10), dpi= 80)
    sns.heatmap(df_reversed_cols, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
    plt.xlabel('Remaining Capacity')
    plt.ylabel('Remaining Time')
    plt.title('Action Map for Lookahead='+str(win))

    patch_0 = mpatches.Patch(color='tab:blue', label='None')
    patch_1 = mpatches.Patch(color='lavender', label='Highest Type')
    patch_2 = mpatches.Patch(color='tab:orange', label='2 Highest Types')
    patch_3 = mpatches.Patch(color='tab:grey', label='3 Highest Types')
    patch_4 = mpatches.Patch(color='tab:red', label='All Types')
    plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], loc='upper right', bbox_to_anchor=(1.12, 1))

    path = path_0 + 'action_map/lookahead'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/action_map_'+str(win)+'.png')
    plt.close()

############################################################################################################
periods_plot = [i for i in np.arange(5, T, 5)]
periods_plot.insert(0, 0)

for t in periods_plot:
    df = pd.DataFrame(sol_dynamic[T-t], columns=['$r_4=.5$', '$r_3=1$', '$r_2=1.5$', '$r_1=2$'])
    # Plotting the heatmap
    plt.figure(figsize=(16,10), dpi= 80)
    sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
    plt.xlabel('Remaining Capacity')
    plt.ylabel('Reward type')
    plt.title('Action Map with Remaining Periods='+str(T-t)+' for Optimal Solution')

    min_patch = mpatches.Patch(color='blue', label='Not selected')
    max_patch = mpatches.Patch(color='red', label='Selected')
    plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

    path = path_0 + 'action_map/optimal'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/action_map_'+str(T-t)+'.png')
    plt.close()
    
    for dix, win in enumerate(windows):   
        df = pd.DataFrame(sol_lookahead[win][T-t], columns=['$r_4=.5$', '$r_3=1$', '$r_2=1.5$', '$r_1=2$'])

        # Plotting the heatmap
        plt.figure(figsize=(16,10), dpi= 80)
        sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
        plt.xlabel('Remaining Capacity')
        plt.ylabel('Reward type')
        plt.title('Action Map with Remaining Periods='+str(T-t)+' for lookahead='+str(win))
        
        min_patch = mpatches.Patch(color='blue', label='Not selected')
        max_patch = mpatches.Patch(color='red', label='Selected')
        plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

        path = path_0 + 'action_map/'+'remaining_period_'+str(T-t)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(win)+'.png')
        plt.close()
        
