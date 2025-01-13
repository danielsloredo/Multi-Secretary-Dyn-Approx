import numpy as np 
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
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

n_sims = 5
horizons = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]
window_size = 4/5
window_size_str = '4_5'
window = {i: int(np.ceil(i**(4/5)))+1 for i in horizons}
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
path = path_0 + 'Figures/4_types/regret_'+window_size_str+'_lookahead/'
path_1 = path_0 + 'Figures/4_types/value_functions/'+window_size_str+'_lookahead/'

result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_msecretary(horizons[-1], horizons[-1], probabilities, rewards, vectors)
val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)
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

with open(path_data+'val_lookahead_t.pkl', 'rb') as pickle_file:
    val_lookahead_t = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_t.pkl', 'rb') as pickle_file:
    sol_lookahead_t = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_t.pkl', 'rb') as pickle_file:
    sol_index_lookahead_t = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead_t.pkl', 'rb') as pickle_file:
    val_eval_lookahead_t = pickle.load(pickle_file)

with open(path_data+'val_lookahead_4_5.pkl', 'rb') as pickle_file:
    val_lookahead_4_5 = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_4_5.pkl', 'rb') as pickle_file:
    sol_lookahead_4_5 = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_4_5.pkl', 'rb') as pickle_file:
    sol_index_lookahead_4_5 = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead_4_5.pkl', 'rb') as pickle_file:
    val_eval_lookahead_4_5 = pickle.load(pickle_file)

with open(path_data+'val_lookahead_2.pkl', 'rb') as pickle_file:
    val_lookahead = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_2.pkl', 'rb') as pickle_file:
    sol_lookahead = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_2.pkl', 'rb') as pickle_file:
    sol_index_lookahead = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead_2.pkl', 'rb') as pickle_file:
    val_eval_lookahead = pickle.load(pickle_file)

'''
if not os.path.exists(path_data):
    os.makedirs(path_data)

periods_offline = [i for i in np.arange(horizons[-1]+1)]
values_offline = ms.simulate_offline_msecretary(horizons[-1], periods_offline, probabilities, rewards, n_types, n_sims, seed = 42)
val_offline = {}
for periods in periods_offline:
    val_offline[periods] = values_offline[periods].mean(axis = 0)

with open(path_data+'val_offline_1.pkl', 'wb') as pickle_file:
    pickle.dump(val_offline, pickle_file)
'''

'''
for horizon in tqdm(horizons):
    capacity = horizon
    (result_lookahead[horizon], 
        val_lookahead[horizon], 
        sol_lookahead[horizon], 
        sol_index_lookahead[horizon], 
        next_value) = ms.dynamic_msecretary_lookahead(
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

with open(path_data+'val_lookahead_1.pkl', 'wb') as pickle_file:
    pickle.dump(val_lookahead, pickle_file)
with open(path_data+'sol_lookahead_1.pkl', 'wb') as pickle_file:
    pickle.dump(sol_lookahead, pickle_file)
with open(path_data+'sol_index_lookahead_1.pkl', 'wb') as pickle_file:
    pickle.dump(sol_index_lookahead, pickle_file)
with open(path_data+'val_eval_lookahead_1.pkl', 'wb') as pickle_file:
    pickle.dump(val_eval_lookahead, pickle_file)
'''

'''
(result_lookahead_4_5, 
    val_lookahead_4_5, 
    sol_lookahead_4_5, 
    sol_index_lookahead_4_5, 
    next_value) = ms.dynamic_msecretary_lookahead_3(
        horizons[-1], 
        horizons[-1], 
        val_deterministic,
        window_size,  
        probabilities, 
        rewards, 
        vectors)
(result_eval_lookahead_4_5, 
    val_eval_lookahead_4_5) = ms.dynamic_evaluate_solution(
        horizons[-1], 
        horizons[-1], 
        sol_lookahead_4_5, 
        probabilities, 
        rewards)

with open(path_data+'val_lookahead_4_5_2.pkl', 'wb') as pickle_file:
    pickle.dump(val_lookahead_4_5, pickle_file)
with open(path_data+'sol_lookahead_4_5_2.pkl', 'wb') as pickle_file:
    pickle.dump(sol_lookahead_4_5, pickle_file)
with open(path_data+'sol_index_lookahead_4_5_2.pkl', 'wb') as pickle_file:
    pickle.dump(sol_index_lookahead_4_5, pickle_file)
with open(path_data+'val_eval_lookahead_4_5_2.pkl', 'wb') as pickle_file:
    pickle.dump(val_eval_lookahead_4_5, pickle_file)
'''

test_bound = {}
test_differences_value = {}
test_bound_3 = {}
for horizon in horizons:
    #regret[horizon] = np.max(val_offline[horizon]-val_eval_lookahead[horizon][horizon][:horizon+1])
    #regret_opt[horizon] = np.max(val_offline[horizon]-val_dynamic[horizon][:horizon+1])
    #sub_opt_gap[horizon] = np.max(val_dynamic[horizon][:horizon+1]-val_eval_lookahead_t[horizon][:horizon+1])
    sub_opt_gap[horizon] = np.max(val_dynamic[horizon][:horizon+1]-val_eval_lookahead_4_5[horizon][:horizon+1])
    
    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(3,horizon+1):
        window = int(np.ceil(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        test_differences[t] = .25*(np.diff(val_deterministic[t][:t+1], n=1)-np.diff(val_dynamic[t][:t+1], n=1))
        for indx, treshold in enumerate(cumsum[:-1]):
            test_differences[t][int(np.floor(treshold-.25*window)):int(np.ceil(treshold+.25*window))+1] = 0
        test_delta[t] = np.absolute(test_differences[t]).max()
    test_bound_3[horizon] = test_delta.sum()

test_bound_2 = {}
for horizon in horizons:
    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(3,horizon+1):
        window = int(np.floor(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        test_differences_1 = .25*(np.diff(val_deterministic[t][:t+1], n=1)-np.diff(val_eval_lookahead_4_5[t][:t+1], n=1))#.25*(np.diff(val_deterministic[t][:t+1], n=1)+val_offline[t][:t]-val_eval_lookahead_4_5[t][1:t+1])
        test_differences_2 = .25*(np.diff(val_eval_lookahead_4_5[t][:t+1], n=1)-np.diff(val_deterministic[t][:t+1], n=1)) #.25*(val_offline[t][1:t+1]-val_eval_lookahead_4_5[t][:t] - np.diff(val_deterministic[t][:t+1], n=1))
        
        #test_differences_1 = .25*(np.diff(val_deterministic[t-1][:t+1], n=1)-(val_eval_lookahead_4_5[t-1][1:t+1]-val_lookahead_4_5[t-1][:t]))
        #test_differences_2 = .25*((val_lookahead_4_5[t-1][1:t+1]-val_eval_lookahead_4_5[t-1][:t])-np.diff(val_deterministic[t-1][:t+1], n=1)) 

        #test_differences_1 = .25*(np.diff(val_deterministic[t][:t+1], n=1)-(val_eval_lookahead_4_5[t][1:t+1]+val_eval_lookahead_4_5[t][:t]-val_lookahead_4_5[t][1:t+1]-val_lookahead_4_5[t][:t]))-.6
        
        if t>3:
            for indx, treshold in enumerate(cumsum[:-1]):
                test_differences[t] = test_differences_1
                test_differences[t][int(np.floor(treshold)):int(np.ceil(cumsum[indx+1]))] = test_differences_2[int(np.floor(treshold)):int(np.ceil(cumsum[indx+1]))]
                test_differences[t][int(np.floor(treshold-.25*window)):int(np.ceil(treshold+.25*window)+1)] = 0
            #test_differences[t][:int(np.floor(.5*.25*window))] = 0
            #test_differences[t][:int(np.floor(cumsum[0]-.25*window))] = 0
            test_differences[t][int(np.floor(cumsum[-2]+.25*window)):] = 0
            test_delta[t] = test_differences[t].max()#np.absolute(test_differences[t]).max() 
    test_bound_2[horizon] = test_delta.sum()

#####################################################################
##### Revision plots
plt.plot(np.diff(np.diff(val_eval_lookahead_4_5[100][:100+1], n=1), n=1), label = 'Heuristic')
plt.legend()
plt.show()

plt.plot(np.absolute(val_deterministic[600][1:600+1]-val_deterministic[600][0:600] - (val_dynamic[600][1:600+1]-val_dynamic[600][:600])))
plt.plot(np.absolute(val_deterministic[600][1:600+1]-val_deterministic[600][0:600] -(val_eval_lookahead_4_5[600][1:600+1]-val_deterministic[600][0:600])))
plt.show()

t=100
plt.plot(np.diff(val_deterministic[t][:t+1], n=1), label = 'Fluid')
plt.plot(np.diff(val_dynamic[t][:t+1], n=1), label = 'DP')
#plt.plot(np.diff(val_eval_test[t][:t+1], n=1), label = 'DP')
plt.plot(np.diff(val_eval_lookahead_4_5[t][:t+1], n=1), label = 'Heuristic')
#plt.plot(val_offline[100][1:101]-val_eval_lookahead_4_5[100][:100], label = 'Bound')
#plt.plot(val_deterministic[100][1:101]-val_eval_lookahead_4_5[100][:100], label = 'Bound')
plt.legend()
plt.show()

#plt.plot(np.diff(val_deterministic[100][:100+1], n=1)-np.diff(val_dynamic[100][:100+1], n=1))
plt.plot(np.maximum(np.diff(val_dynamic[100][:100+1], n=1)-np.diff(val_eval_lookahead_4_5[100][:100+1], n=1), 0))
plt.show()

#plt.plot(val_dynamic[t][:t+1], label= 'DP')
#plt.plot(val_dynamic[t+1][:t+1], label = 'DP2')
plt.plot(val_eval_lookahead_4_5[t][:t+10], label = 'Lookahead')
plt.plot(val_eval_lookahead_4_5[t+1][:t+10], label = 'Lookahead2')
plt.legend()
plt.show()

plt.plot(np.diff(val_dynamic[1000][30:1000+1], n=1)-np.diff(val_eval_lookahead_4_5[1000][30:1000+1], n=1))
plt.show()

plt.plot(val_deterministic[50][:50+1]-val_eval_lookahead_4_5[50][:50+1])
plt.show()

t=100
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[100][t][:t+1], label = '100')
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[150][t][:t+1], label = '150')
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[700][t][:t+1], label = '700')
plt.plot(val_deterministic[t][:t+1]-val_dynamic[t][:t+1], label = 'DP')
plt.legend()
plt.show()
########################################################################


########################################################################
## Action Map 
########################################################################
cmap_dict = {0: 'tab:blue', 1: 'lavender', 2: 'tab:orange', 3: 'tab:grey', 4: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(5)])
df = pd.DataFrame(sol_index_lookahead_4_5[:100, :100+1])
df_reversed_cols = df.iloc[:, ::-1]
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
plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], loc='upper left', bbox_to_anchor=(1.12, 1))
plt.show()
plt.savefig(path_0+'action_map_2_3.png')


########################################################################
## Bounds with the optimal value function first order difference and increasing lookahead 
########################################################################
plt.figure(figsize=(16,10), dpi= 80)
fraction_str = ['2/3', '7/10', '3/4', '4/5', '5/6']
fractions = [2/3, 7/10, 3/4, 4/5, 5/6]
plt.figure(figsize=(16,10), dpi= 80)
for indx_frac, frac in enumerate(fractions):
    for horizon in horizons:
        
        test_delta = np.zeros(horizon+1)
        test_differences = {}
        for t in range(3,horizon+1):
            window = int(np.ceil(t**(frac)))+1
            restriction = np.multiply(probabilities[::-1], t)
            cumsum = np.cumsum(restriction)
            test_differences[t] = .25*(np.diff(val_deterministic[t][:t+1], n=1)-np.diff(val_dynamic[t][:t+1], n=1))
            for indx, treshold in enumerate(cumsum[:-1]):
                test_differences[t][int(np.floor(treshold-.25*window)):int(np.ceil(treshold+.25*window))+1] = 0
            test_delta[t] = np.absolute(test_differences[t]).max()
        test_bound_3[horizon] = test_delta.sum()
    sorted_data = dict(sorted(test_bound_3.items()))
    x = list(sorted_data.keys())
    y = list(sorted_data.values())
    plt.plot(x, y, color='black', marker='', markersize=5, markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 't^'+fraction_str[indx_frac])
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Bound Sum of Max |Deltas| on ND', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()
plt.close()

########################################################################
## Bounds on regret
########################################################################
if not os.path.exists(path):
    os.makedirs(path)

plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(sub_opt_gap.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Regret')
sorted_data = dict(sorted(test_bound_3.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='black', marker='', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Sum of max |Deltas| on ND (Optimal)')
sorted_data = dict(sorted(test_bound_2.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:orange', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:orange', label = 'Bound')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Regret for $t^{4/5}$ Lookahead Policy', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()

########################################################################
## Value functions and differences between value functions 
########################################################################
if not os.path.exists(path_1):
    os.makedirs(path_1)

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
    plt.plot(val_lookahead_t[fix_t][:fix_t+1]-val_dynamic[fix_t][:fix_t+1], color = 'tab:blue', label='Difference Lookahead DP-DP', marker='o', fillstyle = 'none')
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
    plt.savefig(path_1+'/diff/diff_value_functions'+str(fix_t)+'_new.png')
    plt.close()


############################################################################
#Plots for Presentation
############################################################################

path_presentation = path_0 + 'Figures/presentation/'
if not os.path.exists(path_presentation):
    os.makedirs(path_presentation)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(np.diff(val_dynamic[100][:100+1]), color = 'tab:blue', linestyle = '-', marker = '.', fillstyle = 'none', label='DP')
plt.axhline(y=2, color='black', linestyle='-', linewidth=1)
plt.text(x=3, y=2.02, s='r(1)', color='black', fontsize=12, va='center')
plt.axhline(y=1.5, color='black', linestyle='-', linewidth=1)
plt.text(x=5, y=1.52, s='r(2)', color='black', fontsize=12, va='center')
plt.axhline(y=1, color='black', linestyle='-', linewidth=1)
plt.text(x= 5, y=1.02, s='r(3)', color='black', fontsize=12, va='center')
plt.axhline(y=0.5, color='black', linestyle='-', linewidth=1)
plt.text(x=5, y=0.52, s='r(4)', color='black', fontsize=12, va='center')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.xlim(left=0)
plt.xlabel('K (budget)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.savefig(path_presentation+'shadow_costs_dp.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(np.diff(val_dynamic[100][:100+1]), color = 'tab:blue', linestyle = '-', marker = '.', fillstyle = 'none', label='Optimal DP')
plt.plot(np.diff(val_deterministic[100][:100+1]), color = 'tab:red', linestyle = '-', marker = '', fillstyle = 'none', label='Deterministic UB')
plt.axhline(y=2, color='black', linestyle='-', linewidth=.5)
plt.text(x=3, y=2.02, s='r(1)', color='black', fontsize=12, va='center')
plt.axhline(y=1.5, color='black', linestyle='-', linewidth=.5)
plt.text(x=5, y=1.52, s='r(2)', color='black', fontsize=12, va='center')
plt.axhline(y=1, color='black', linestyle='-', linewidth=.5)
plt.text(x= 5, y=1.02, s='r(3)', color='black', fontsize=12, va='center')
plt.axhline(y=0.5, color='black', linestyle='-', linewidth=.5)
plt.text(x=5, y=0.52, s='r(4)', color='black', fontsize=12, va='center')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.xlim(left=0)
plt.xlabel('k (budget)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(loc = "upper right")   
plt.savefig(path_presentation+'shadow_costs_dp_fluid.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(val_dynamic[100][:100+1], color = 'tab:blue', label='Optimal Value Function', linestyle = '-', marker = '.', fillstyle = 'none')   
plt.plot(val_deterministic[100][:100+1], color = 'tab:red', label='Deterministic UB')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('k (budget)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'value_dp_fluid.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(val_deterministic[100][:100+1] - val_dynamic[100][:100+1], color = 'tab:blue', linestyle = '-', marker = '.', fillstyle = 'none')   
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('k (budget)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.savefig(path_presentation+'diff_value_dp_fluid.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(np.diff(val_dynamic[100][:100+1]), color = 'tab:blue', linestyle = '-', marker = '.', fillstyle = 'none', label='Optimal DP')
plt.plot(np.diff(val_deterministic[100][:100+1]), color = 'tab:red', linestyle = '-', marker = '', fillstyle = 'none', label='Deterministic UB')
plt.plot(np.diff(val_lookahead_4_5[100][:100+1]), color = 'tab:gray', linestyle = '-', marker = '', fillstyle = 'none', label='Lookahead Approximation')
plt.axhline(y=2, color='black', linestyle='-', linewidth=.5)
plt.text(x=3, y=2.02, s='r(1)', color='black', fontsize=12, va='center')
plt.axhline(y=1.5, color='black', linestyle='-', linewidth=.5)
plt.text(x=5, y=1.52, s='r(2)', color='black', fontsize=12, va='center')
plt.axhline(y=1, color='black', linestyle='-', linewidth=.5)
plt.text(x= 5, y=1.02, s='r(3)', color='black', fontsize=12, va='center')
plt.axhline(y=0.5, color='black', linestyle='-', linewidth=.5)
plt.text(x=5, y=0.52, s='r(4)', color='black', fontsize=12, va='center')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.xlim(left=0)
plt.xlabel('k (budget)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(loc = "upper right")   
plt.savefig(path_presentation+'shadow_costs_dp_fluid_lookahead.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(sub_opt_gap.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Regret')
sorted_data = dict(sorted(test_bound_3.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='black', marker='', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Bound with optimal value')
sorted_data = dict(sorted(test_bound_2.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:orange', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:orange', label = 'Bound with lookahead value')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.savefig(path_presentation+'regret_bounds.png')
plt.close()