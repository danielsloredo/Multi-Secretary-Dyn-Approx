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
probabilities = np.array([.1, .6, .05, .25])#np.array([.2, .3, .15, .35])#np.array([.5, .5]) 
rewards = 100000*np.array([.5, 1, 1.5, 2])# np.array([1, 2]) #

vectors = ms.generate_vectors(n_types)
prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.

n_sims = 5
horizons = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]#, 2500, 3000, 4000, 5000]#, 10000]
window_size = 2/3
window_size_str = '2_3'
window = {i: int(i**(window_size))+1 for i in horizons}#{i: int(i*.25) + 1 for i in horizons}
########

path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'
path_data = path_0 + 'Data/4_types/regret_2_3/'
path_offline = path_0 + 'Data/4_types/'
path = path_0 + 'Figures/4_types/regret_'+window_size_str+'_lookahead/'
path_1 = path_0 + 'Figures/4_types/value_functions/'+window_size_str+'_lookahead/'

'''
result_dynamic, val_dynamic, sol_dynamic, sol_index_dynamic = ms.dynamic_msecretary(horizons[-1], horizons[-1], probabilities, rewards, vectors)
val_deterministic = ms.deterministic_msecretary_array(horizons[-1], horizons[-1], probabilities, rewards, n_types)

with open(path_data+'val_dynamic.pkl', 'wb') as pickle_file:
    pickle.dump(val_dynamic, pickle_file)
with open(path_data+'sol_dynamic.pkl', 'wb') as pickle_file:
    pickle.dump(sol_dynamic, pickle_file)
with open(path_data+'sol_index_dynamic.pkl', 'wb') as pickle_file:
    pickle.dump(sol_index_dynamic, pickle_file)
with open(path_data+'val_deterministic.pkl', 'wb') as pickle_file:
    pickle.dump(val_deterministic, pickle_file)
'''

#values_offline = ms.simulate_offline_msecretary(horizons[-1], horizons, probabilities, rewards, n_types, n_sims, seed = 42)
#for horizon in horizons:
#    val_offline[horizon] = values_offline[horizon].mean(axis = 0)

with open(path_data+'val_deterministic.pkl', 'rb') as pickle_file:
    val_deterministic = pickle.load(pickle_file)

with open(path_data+'val_dynamic.pkl', 'rb') as pickle_file:
    val_dynamic = pickle.load(pickle_file)
with open(path_data+'sol_dynamic.pkl', 'rb') as pickle_file:
    sol_dynamic = pickle.load(pickle_file)
with open(path_data+'sol_index_dynamic.pkl', 'rb') as pickle_file:
    sol_index_dynamic = pickle.load(pickle_file)

with open(path_data+'val_lookahead_batched.pkl', 'rb') as pickle_file:
    val_lookahead_batched = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_batched.pkl', 'rb') as pickle_file:
    sol_lookahead_batched = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_batched.pkl', 'rb') as pickle_file:
    sol_index_lookahead_batched = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead_batched.pkl', 'rb') as pickle_file:
    val_eval_lookahead_batched = pickle.load(pickle_file)

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

with open(path_data+'val_lookahead_4_5_2.pkl', 'rb') as pickle_file:
    val_lookahead_4_5_2 = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_4_5_2.pkl', 'rb') as pickle_file:
    sol_lookahead_4_5_2 = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_4_5_2.pkl', 'rb') as pickle_file:
    sol_index_lookahead_4_5_2 = pickle.load(pickle_file)
with open(path_data+'val_eval_lookahead_4_5_2.pkl', 'rb') as pickle_file:
    val_eval_lookahead_4_5_2 = pickle.load(pickle_file)

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

result_lookahead = {}
val_lookahead = {}
sol_lookahead = {}
sol_index_lookahead = {}
result_eval_lookahead = {}
val_eval_lookahead = {}

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
### Batched Lookahead 
result_lookahead_batched = {}
val_lookahead_batched = {}
sol_lookahead_batched = {}
sol_index_lookahead_batched = {}
result_eval_lookahead_batched = {}
val_eval_lookahead_batched = {}
indifference_lookahead = {}

#horizons = [10000]

for horizon in tqdm(horizons):
    capacity = horizon
    (val_lookahead_batched[horizon], 
        sol_lookahead_batched[horizon], 
        sol_index_lookahead_batched[horizon],
        indifference_lookahead[horizon]) = ms.dynamic_msecretary_lookahead_batched(
            horizon, 
            capacity, 
            val_deterministic, 
            window[horizon], 
            probabilities, 
            rewards, 
            vectors)
    (result_eval_lookahead_batched[horizon], 
        val_eval_lookahead_batched[horizon]) = ms.dynamic_evaluate_solution(
            horizon, 
            capacity, 
            sol_lookahead_batched[horizon],
            indifference_lookahead[horizon], 
            probabilities, 
            rewards)
'''    
with open(path_data+'val_lookahead_batched_random.pkl', 'wb') as pickle_file:
    pickle.dump(val_lookahead_batched, pickle_file)
with open(path_data+'sol_lookahead_batched_random.pkl', 'wb') as pickle_file:
    pickle.dump(sol_lookahead_batched, pickle_file)
with open(path_data+'sol_index_lookahead_batched_random.pkl', 'wb') as pickle_file:
    pickle.dump(sol_index_lookahead_batched, pickle_file)
with open(path_data+'val_eval_lookahead_batched_random.pkl', 'wb') as pickle_file:
    pickle.dump(val_eval_lookahead_batched, pickle_file)
'''

###################################################################
#Deviation compensation

def deviation_compensation(t, x, sol, probabilities, rewards, value_function, bellman = True):
    next_less = value_function[t-1, x-1]
    next_same = value_function[t-1, x]
    if bellman:
        logic_test = (rewards + next_less >= next_same)
    else:
        logic_test = sol[t, x] > 0 
    q_val = np.where(logic_test, rewards + next_less, next_same)
    compensation = value_function[t,x] - np.sum(np.multiply(probabilities, q_val))
    return compensation


val_lookahead_deltas = {}
sol_lookahead_deltas = {}
sol_index_lookahead_deltas = {}
val_next_lookahead_deltas = {}

val_lookahead_period = {}
sol_lookahead_period = {}
sol_index_lookahead_period = {}
val_next_lookahead_period = {}

delta_eval = {}
delta_bellman = {}
delta_eval_temp = {}
delta_bellman_temp = {}

delta_eval_sum = {}
delta_bellman_sum = {}

period = 1500
T=2000
window_values = np.linspace(0,1,11)
window_str = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
for indx, window in enumerate(window_values): 
    (val_lookahead_period[window_str[indx]], 
     sol_lookahead_period[window_str[indx]],
     sol_index_lookahead_period[window_str[indx]], 
     val_next_lookahead_period[window_str[indx]]
     )  = ms.dynamic_lookahead_period(period, 
                                        T, 
                                        val_deterministic, 
                                        window, 
                                        probabilities, 
                                        rewards, 
                                        vectors)
    delta_eval_temp[window_str[indx]] = np.zeros(T+1)
    delta_bellman_temp[window_str[indx]] = np.zeros(T+1)
    for x in range(1, T+1):
        delta_eval_temp[window_str[indx]][x] = deviation_compensation(period, x, sol_lookahead_period[window_str[indx]], probabilities, rewards, val_lookahead_period[window_str[indx]], bellman = False)
        delta_bellman_temp[window_str[indx]][x] = deviation_compensation(period, x, sol_lookahead_period[window_str[indx]], probabilities, rewards, val_lookahead_period[window_str[indx]], bellman = True)
    #print(delta_bellman_temp)
    delta_eval[window] = np.absolute(delta_eval_temp[window_str[indx]]/100000).max()
    delta_bellman[window] = np.absolute(delta_bellman_temp[window_str[indx]]/100000).max()

for indx, window in enumerate(window_values): 
    '''
    (res,
     val_lookahead_deltas[window_str[indx]], 
     sol_lookahead_deltas[window_str[indx]],
     sol_index_lookahead_deltas[window_str[indx]], 
     val_next_lookahead_deltas[window_str[indx]]
     )  = ms.dynamic_msecretary_lookahead_3(period, 
                                        T, 
                                        val_deterministic, 
                                        window, 
                                        probabilities, 
                                        rewards, 
                                        vectors)
    '''
    test_delta = np.zeros((period+1,T+1))
    delta_eval_temp_sum = np.zeros(period+1)
    test_delta_bellman = np.zeros((period+1,T+1))
    delta_bellman_temp_sum = np.zeros(period+1)
    delta_action = np.zeros(period+1)
    delta_action_bellman = np.zeros(period+1)
    for t in range(1,period+1):
        for x in range(1, t+1):
            test_delta[t, x] = deviation_compensation(t, x, sol_lookahead_deltas[window_str[indx]], probabilities, rewards, val_lookahead_deltas[window_str[indx]], bellman = False)
            test_delta_bellman[t, x] = deviation_compensation(t, x, sol_lookahead_deltas[window_str[indx]], probabilities, rewards, val_lookahead_deltas[window_str[indx]], bellman = True)
        delta_eval_temp_sum[t] = np.absolute(test_delta[t]).max()
        delta_bellman_temp_sum[t] = np.absolute(test_delta_bellman[t]).max()
        delta_action[t] = np.absolute(test_delta[t]).argmax()
        delta_action_bellman[t] = np.absolute(test_delta_bellman[t]).argmax()
    delta_eval_sum[window_str[indx]] = delta_eval_temp_sum.sum()/100000
    delta_bellman_sum[window_str[indx]] = delta_bellman_temp_sum.sum()/100000
'''
with open(path_data+'val_lookahead_deltas_1500.pkl', 'wb') as pickle_file:
    pickle.dump(val_lookahead_deltas, pickle_file)
with open(path_data+'sol_lookahead_deltas_1500.pkl', 'wb') as pickle_file:
    pickle.dump(sol_lookahead_deltas, pickle_file)
with open(path_data+'sol_index_lookahead_deltas_1500.pkl', 'wb') as pickle_file:
    pickle.dump(sol_index_lookahead_deltas, pickle_file)
with open(path_data+'val_next_lookahead_deltas_1500.pkl', 'wb') as pickle_file:
    pickle.dump(val_next_lookahead_deltas, pickle_file)
'''

with open(path_data+'val_lookahead_deltas.pkl', 'rb') as pickle_file:
    val_lookahead_deltas = pickle.load(pickle_file)
with open(path_data+'sol_lookahead_deltas.pkl', 'rb') as pickle_file:
    sol_lookahead_deltas = pickle.load(pickle_file)
with open(path_data+'sol_index_lookahead_deltas.pkl', 'rb') as pickle_file:
    sol_index_lookahead_deltas = pickle.load(pickle_file)
with open(path_data+'val_next_lookahead_deltas.pkl', 'rb') as pickle_file:
    val_eval_lookahead_deltas = pickle.load(pickle_file)

plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(delta_bellman_sum.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Sum of Deltas with max action')
sorted_data = dict(sorted(delta_eval_sum.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='black', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Sum of Deltas with used action')
plt.title('Sum of Deltas for t=1500 in problem with horizon T=2000', fontsize=20)
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Exponent in T^a lookahead steps', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(delta_bellman.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Delta with max action')
sorted_data = dict(sorted(delta_eval.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='black', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Delta with used action')
plt.title('Delta for t=1500 in problem with horizon T=2000', fontsize=20)
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Exponent in T^a lookahead steps', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(delta_bellman_temp['0.6'][:period+1], color='black', marker='x', markersize=2, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Delta with T^{2/3} lookahead')
plt.plot(delta_bellman_temp['0.5'][:period+1], color='tab:red', linewidth = .5, marker='o', markersize=2, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Delta with T^{1/2} lookahead')
plt.title('Delta for t=1500 in problem with horizon T=2000', fontsize=20)
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('x', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()

###################################################################
###################################################################
#Lookahead Evolution
###################################################################
period = 1000
capacity=2000
steps = 11

(val_lookahead_step, 
     sol_lookahead_step,
     sol_index_lookahead_step 
     )  = ms.dynamic_lookahead_step(period, 
                                        capacity, 
                                        val_deterministic, 
                                        steps, 
                                        probabilities, 
                                        rewards, 
                                        vectors)

(val_lookahead_step_h, 
     sol_lookahead_step_h,
     sol_index_lookahead_step_h 
     )  = ms.dynamic_lookahead_step_homogeneous(period, 
                                        capacity, 
                                        val_deterministic, 
                                        4, 
                                        probabilities, 
                                        rewards, 
                                        vectors)


(val_lookahead_step_f, 
     sol_lookahead_step_f,
     sol_index_lookahead_step_f 
     )  = ms.dynamic_lookahead_step_forward(period, 
                                        capacity, 
                                        val_deterministic, 
                                        4, 
                                        probabilities, 
                                        rewards, 
                                        vectors)


colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7"]  
line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # The last one is a custom dash pattern

plt.figure(figsize=(16,10), dpi= 80)
for index, t in enumerate(range(period-steps, period+1)):
    plt.plot(val_lookahead_step[t, :period+10]/100000, label = 't='+str(t), color = 'black', linestyle = line_styles[index])
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()


plt.figure(figsize=(16,10), dpi=80)
x_values = np.array([x/period for x in range(capacity+1)])  # Normalize x values between 0 and 1
for index, t in enumerate(range(period-steps, period+1)):
    plt.plot(x_values[1:period+11], val_lookahead_step_h[t, :period+10]/t/100000, 
             label=f't={t}', color='black', linestyle=line_styles[index])
plt.xticks(fontsize=12, alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.legend()
plt.xlabel('x/t', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()



plt.figure(figsize=(16,10), dpi= 80)
for step in range(1, steps+1):
    plt.plot(val_lookahead_step[step, :period+1]/100000, label = 'steps='+str(step), color = 'black', linestyle = line_styles[step])
plt.plot(val_deterministic[period, :period+1]/100000, label = 'Fluid', color = 'black', linestyle = line_styles[0])
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()


cmap_dict = {0: 'tab:blue', 1: 'lavender', 2: 'tab:orange', 3: 'tab:grey', 4: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(5)])
df = pd.DataFrame(sol_index_lookahead_step[period-steps+1:period+1, :period+1])
df_reversed_cols = df
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed_cols, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
plt.xlabel('x')
custom_y_labels = np.array([i for i in range(period-steps+1, period+1)])  # Example: Change these values as needed
plt.yticks(ticks=np.arange(df.shape[0]), labels=custom_y_labels.astype(int), rotation=0)
plt.ylabel('t')
plt.title('Action Map for Optimal Solution')
patch_0 = mpatches.Patch(color='tab:blue', label='None')
patch_1 = mpatches.Patch(color='lavender', label='Highest Type')
patch_2 = mpatches.Patch(color='tab:orange', label='2 Highest Types')
patch_3 = mpatches.Patch(color='tab:grey', label='3 Highest Types')
patch_4 = mpatches.Patch(color='tab:red', label='All Types')
plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], loc='upper right')
plt.show()


cmap_dict = {0: 'tab:orange', 1: 'tab:grey'}
cmap = ListedColormap([cmap_dict[i] for i in range(2)])
df = pd.DataFrame(sol_index_lookahead_step[period-steps+1:period+1, 275:325+1])
df_reversed_cols = df
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed_cols, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
custom_x_labels = np.array([i for i in range(275, 325+1)])  # Example: Change these values as needed
plt.xticks(ticks=np.arange(df.shape[1]), labels=custom_x_labels.astype(int), rotation=90)
plt.xlabel('x')
custom_y_labels = np.array([i for i in range(period-steps+1, period+1)])  # Example: Change these values as needed
plt.yticks(ticks=np.arange(df.shape[0]), labels=custom_y_labels.astype(int), rotation=0)
plt.ylabel('t')
plt.title('Action Map for Optimal Solution')
patch_0 = mpatches.Patch(color='tab:blue', label='None')
patch_1 = mpatches.Patch(color='lavender', label='Highest Type')
patch_2 = mpatches.Patch(color='tab:orange', label='2 Highest Types')
patch_3 = mpatches.Patch(color='tab:grey', label='3 Highest Types')
patch_4 = mpatches.Patch(color='tab:red', label='All Types')
plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], loc='upper right')
plt.show()

###################################################################
#Bounds
regret = {}
regret_opt = {}
val_offline = {}
sub_opt_gap = {}


test_differences_value = {}
test_bound = {}

test_bound_3 = {}

window_size = 2/3
window = {i: int(i**(window_size))+1 for i in horizons}

val_eval_lookahead_bounds = np.copy(val_eval_lookahead_t)
val_lookahead_bounds = np.copy(val_lookahead_t)
sol_lookahead_bounds = np.copy(sol_lookahead_t)

for horizon in horizons:
    #regret[horizon] = np.max(val_offline[horizon]-val_eval_lookahead_bounds[horizon][horizon][:horizon+1])
    #regret_opt[horizon] = np.max(val_offline[horizon]-val_dynamic[horizon, :horizon+1])
    sub_opt_gap[horizon] = np.max(val_dynamic[horizon, :horizon+1]/100000-val_eval_lookahead_bounds[horizon, :horizon+1])
    
    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(3,horizon+1):
        window_t = int(np.ceil(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        test_differences[t] = .25*(np.diff(val_deterministic[t, :t+1], n=1)-np.diff(val_dynamic[t, :t+1], n=1))
        for indx, treshold in enumerate(cumsum[:-1]):
            test_differences[t][int(np.floor(treshold-.25*window_t)):int(np.ceil(treshold+.25*window_t))+1] = 0
        test_delta[t] = np.absolute(test_differences[t]).max()
    test_bound_3[horizon] = test_delta.sum()


for horizon in horizons:
    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(3,horizon+1):
        window_t = int(np.floor(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        test_differences_1 = .25*(np.diff(val_deterministic[t, :t+1], n=1)-np.diff(val_eval_lookahead_bounds[t, :t+1], n=1))#.25*(np.diff(val_deterministic[t, :t+1], n=1)+val_offline[t, :t]-val_eval_lookahead_4_5[t, 1:t+1])
        test_differences_2 = .25*(np.diff(val_eval_lookahead_bounds[t, :t+1], n=1)-np.diff(val_deterministic[t, :t+1], n=1)) #.25*(val_offline[t, 1:t+1]-val_eval_lookahead_4_5[t, :t] - np.diff(val_deterministic[t, :t+1], n=1))
        #test_differences_1 = .25*(np.diff(val_deterministic[t-1, :t+1], n=1)-(val_eval_lookahead_4_5[t-1, 1:t+1]-val_lookahead_4_5[t-1, :t]))
        #test_differences_2 = .25*((val_lookahead_4_5[t-1, 1:t+1]-val_eval_lookahead_4_5[t-1, :t])-np.diff(val_deterministic[t-1, :t+1], n=1)) 
        #test_differences_1 = .25*(np.diff(val_deterministic[t, :t+1], n=1)-(val_eval_lookahead_4_5[t, 1:t+1]+val_eval_lookahead_4_5[t, :t]-val_lookahead_4_5[t, 1:t+1]-val_lookahead_4_5[t, :t]))-.6        
        if t>3:
            for indx, treshold in enumerate(cumsum[:-1]):
                test_differences[t] = test_differences_1
                test_differences[t][int(np.floor(treshold)):int(np.ceil(cumsum[indx+1]))] = test_differences_2[int(np.floor(treshold)):int(np.ceil(cumsum[indx+1]))]
                test_differences[t][int(np.floor(treshold-.25*window_t)):int(np.ceil(treshold+.25*window_t)+1)] = 0
            #test_differences[t][:int(np.floor(.5*.25*window_t))] = 0
            test_differences[t][:int(np.floor(cumsum[0]-.25*window_t))] = 0
            #test_differences[t][int(np.floor(cumsum[-2]+.25*window_t)):] = 0
            test_delta[t] = test_differences[t].max()
    test_bound[horizon] = test_delta.sum()

test_bound_4_5 = {}
for horizon in horizons:
    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(10,horizon+1):
        window_t = int(np.floor(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        compensations = np.zeros(t+1)
        for x in range(1,t+1):
            compensations[x] = deviation_compensation(t, x, sol_lookahead_bounds, probabilities, rewards, val_lookahead_bounds)
        test_differences[t] = compensations
        #for indx, treshold in enumerate(cumsum[:-1]):
        #    test_differences[t][int(np.ceil(treshold-.25*window_t))+1:int(np.floor(treshold+.25*window_t))] = 0
        #test_differences[t][int(np.floor(cumsum[-2]+.25*window_t)):] = 0
        #test_differences[t][:int(np.floor(cumsum[0]-.25*window_t))] = 0
        test_delta[t] = test_differences[t][1:-1].max() 
    test_bound_4_5[horizon] = test_delta.sum()

test_bound[0] = 0 
test_bound_3[0] = 0 
test_bound_4_5[0] = 0

#####################################################################
#batched lookahead bounds
sub_opt_gap_batched = {}
test_bound_optimal_batched = {}
gap_approx = {}
gap_fluid = {}
gap_v0_v1 = {}

for horizon in horizons:
    sub_opt_gap_batched[horizon] = np.max(val_dynamic[horizon, :horizon+1]-val_eval_lookahead_batched[horizon][horizon, :horizon+1])/100000
    gap_approx[horizon] = np.max(val_lookahead_batched[horizon][horizon, :horizon+1]-val_dynamic[horizon, :horizon+1])/100000
    gap_fluid[horizon] = np.max(val_deterministic[horizon, :horizon+1]-val_dynamic[horizon, :horizon+1])/100000
    gap_v0_v1[horizon] = np.max(val_lookahead_batched[horizon][horizon, :horizon+1]-val_eval_lookahead_batched[horizon][horizon, :horizon+1])/100000

    test_delta = np.zeros(horizon+1)
    test_differences = {}
    for t in range(3,horizon+1):
        window_t = int(np.ceil(t**(window_size)))+1
        restriction = np.multiply(probabilities[::-1], t)
        cumsum = np.cumsum(restriction)
        test_differences[t] = .25*(np.diff(val_deterministic[t, :t+1], n=1)-np.diff(val_dynamic[t, :t+1], n=1))
        for indx, treshold in enumerate(cumsum[:-1]):
            test_differences[t][int(np.floor(treshold-.25*window_t)):int(np.ceil(treshold+.25*window_t))+1] = 0
        test_delta[t] = np.absolute(test_differences[t]).max()
    test_bound_optimal_batched[horizon] = test_delta.sum()

test_bound_4_5_batched = {}
deltas = {}
for horizon in horizons:
    test_delta = np.zeros((horizon+1,horizon+1))
    delta_use = np.zeros(horizon+1)
    for t in range(10,horizon+1):
        for x in range(1, t+1):
            test_delta[t, x] = deviation_compensation(t, x, sol_lookahead_batched[horizon], probabilities, rewards, val_lookahead_batched[horizon])
        delta_use[t] = test_delta[t].mean() 
    test_bound_4_5_batched[horizon] = delta_use.sum()
    deltas[horizon] = test_delta


def online_bound(T, periods, probabilities, rewards, n_types, n_sims, seed = 42):
    val_offline = {i : np.zeros((n_sims, i+1)) for i in periods}
    np.random.seed(seed)
    for sim in tqdm(np.arange(n_sims)):
        uniform_1 = np.random.rand(T) 
        arrival_type_1 = np.empty(T, dtype=int)
        uniform_2 = 1-uniform_1 
        arrival_type_2 = np.empty(T, dtype=int)
        for type in np.arange(n_types):
            arrival_type_1[(uniform_1>=np.sum(probabilities[:type])) & (uniform_1 < np.sum(probabilities[:type+1]))] = type
            arrival_type_2[(uniform_2>=np.sum(probabilities[:type])) & (uniform_2 < np.sum(probabilities[:type+1]))] = type
        
        for period in periods:
            number_type_1 = np.zeros(n_types)
            number_type_2 = np.zeros(n_types)
            for type in np.arange(n_types):
                number_type_1[type] = np.sum(arrival_type_1[:period] == type)
                number_type_2[type] = np.sum(arrival_type_2[:period] == type)
            for x in range(1, period+1):
                val_1 = offline_msecretary(number_type_1, rewards, n_types, x)
                val_2 = offline_msecretary(number_type_2, rewards, n_types, x)
                val_offline[period][sim, x] = (val_1 + val_2)/2

    return val_offline


#####################################################################
##### Revision plots

t = 2000
plt.plot(np.diff(np.diff(val_dynamic[t][:t+1]/100000, n=1), n=1), label = 'Optimal')
plt.plot(np.diff(np.diff(val_eval_lookahead_batched[t][t, :t+1]/100000, n=1), n=1), label = 'Heuristic')
plt.legend()
plt.show()

plt.plot(np.diff(val_dynamic[t][:t+1]/100000, n=1), label = 'Optimal')
plt.plot(np.diff(val_eval_lookahead_batched[t][t, :t+1]/100000, n=1), label = 'Heuristic')
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

t=100
plt.plot(val_dynamic[t,:t+1], label= 'DP')
plt.plot(val_lookahead_batched[t][t, :t+1], label = 'Approx')
plt.plot(val_eval_lookahead_batched[t][t,:t+1], label = 'Lookahead')
plt.legend()
plt.show()
t = 3000
plt.plot((val_dynamic[t, :t+1]-val_eval_lookahead_batched[t][t,:t+1])/100000)
plt.show()
t = 3000
plt.plot(val_dynamic[t, :t+1]/100000-val_eval_lookahead_t[t,:t+1])
plt.show()
plt.plot(val_lookahead_batched[t][t, :t+1]-val_dynamic[t,:t+1])
plt.show()


plt.plot(np.diff(val_dynamic[t, :t+1], n=1)-np.diff(val_eval_lookahead_batched[t][t, :t+1], n=1))
plt.show()



t=100
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[100][t][:t+1], label = '100')
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[150][t][:t+1], label = '150')
plt.plot(val_deterministic[t][:t+1]-val_eval_lookahead[700][t][:t+1], label = '700')
plt.plot(val_deterministic[t][:t+1]-val_dynamic[t][:t+1], label = 'DP')
plt.legend()
plt.show()


plt.plot(val_deterministic[:500, 100])
plt.show()
########################################################################


########################################################################
## Action Map 
########################################################################
t = 100
cmap_dict = {0: 'tab:blue', 1: 'lavender', 2: 'tab:orange', 3: 'tab:grey', 4: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(5)])
df = pd.DataFrame(sol_index_dynamic[:t, :t+1])
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
#plt.savefig(path_0+'action_map_2_3.png')

cmap_dict = {0: 'tab:blue', 1: 'lavender', 2: 'tab:orange', 3: 'tab:grey', 4: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(5)])
df = pd.DataFrame(sol_index_lookahead_batched[t][:t, :t+1])
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


########################################################################
## Bounds with the optimal value function first order difference and increasing lookahead 
########################################################################
plt.figure(figsize=(16,10), dpi= 80)
fraction_str = ['2/3', '4/5']
fractions = [2/3, 4/5]
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
## Bounds on regret for the batched lookahead
########################################################################
plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(sub_opt_gap_batched.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Regret')
sorted_data = dict(sorted(gap_approx.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Gap Approximation')
sorted_data = dict(sorted(gap_fluid.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='*', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Gap v0 v1')
#sorted_data = dict(sorted(test_bound_optimal_batched.items()))
#x = list(sorted_data.keys())
#y = list(sorted_data.values())
#plt.plot(x, y, color='black', marker='', markersize=5, 
#        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Using Optimal function bound')
#sorted_data = dict(sorted(test_bound_4_5_batched.items()))
#x = list(sorted_data.keys())
#y = list(sorted_data.values())
#plt.plot(x, y, color='black', marker='x', markersize=5, 
#        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Optimality test bound')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()


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
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Regret >')
sorted_data = dict(sorted(test_bound_3.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='black', marker='', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Sum of max |Deltas| on ND (Optimal)')
sorted_data = dict(sorted(test_bound.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:orange', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:orange', label = 'Bound >')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
#plt.title('Regret for $t^{4/5}$ Lookahead Policy', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()


plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(sub_opt_gap.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:red', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red', label = 'Regret')
#sorted_data = dict(sorted(test_bound_3.items()))
#x = list(sorted_data.keys())
#y = list(sorted_data.values())
#plt.plot(x, y, color='black', marker='', markersize=5, 
#        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='black', label = 'Using Optimal function bound')
sorted_data = dict(sorted(test_bound_4_5.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:green', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:green', label = 'Optimality test bound')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
#plt.title('Regret for $t^{4/5}$ Lookahead Policy', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('Horizon', fontsize = 14)
plt.legend()
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
sorted_data = dict(sorted(test_bound_2_3.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:blue', marker='x', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:blue', label = 'Optimality test bound 2/3')
sorted_data = dict(sorted(test_bound_4_5.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:green', marker='o', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:green', label = 'Optimality test bound 4/5 Conservative')
sorted_data = dict(sorted(test_bound_4_5_2.items()))
x = list(sorted_data.keys())
y = list(sorted_data.values())
plt.plot(x, y, color='tab:green', marker='+', markersize=5, 
        markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:green', label = 'Optimality test bound 4/5 Agressive')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
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