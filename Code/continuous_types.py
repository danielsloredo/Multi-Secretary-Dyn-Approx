import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import seaborn as sns
import pandas as pd
import pickle


def deterministic_continuous_msecretary(t, x):
    #Fluid of the multisecretary problem with continuous types Uniform[0,1]
    if x > t:
        result = t/2
        sol = 0
    else:
        result = x - (x**2)/(2*t)
        sol = 1-x/t
    return result, sol

def deterministic_continuous_msecretary_array(T, capacity):
    #Solve the deterministic version of the multi-secretary problem for all periods and all capacities
    val_deterministic = np.zeros((T+1, capacity+1))
    sol_deterministic = np.zeros((T+1, capacity+1))
    for t in tqdm(range(1, T+1)):
        for x in range(1, capacity+1):
            val_deterministic[t, x], sol_deterministic[t,x] = deterministic_continuous_msecretary(t, x)

    return val_deterministic, sol_deterministic


def dynamic_continuous_msecretary(T, capacity):
    val = np.zeros((T+1, capacity+1))
    sol = np.zeros((T+1, capacity+1))

    for t in tqdm(np.arange(1, T+1)):
        for x in np.arange(1, capacity+1):
            next_less = val[t-1, x-1]
            next_same = val[t-1, x]

            action = next_same - next_less
            if action >= 1:
                sol[t, x] = 1
            elif action <= 0:
                sol[t, x] = 0
            else:
                sol[t, x] = action

            val[t, x] = (1-sol[t,x]**2)/2 + sol[t,x]*next_same + (1-sol[t,x])*next_less

    return val, sol

def dynamic_continuous_msecretary_lookahead(T, capacity, val_deterministic, window_size):
    value = np.zeros((T+1, capacity+1))
    value_next = np.zeros((T+1, capacity+1))
    solution = np.zeros((T+1, capacity+1))

    for period in tqdm(range(1, T+1)): 
        val_temp = np.zeros((T+1, capacity+1))
        sol_temp = np.zeros((T+1, capacity+1))
        if type(window_size) == int:
            window = window_size
        elif window_size == 'log':
            window = int(np.log(period)) + 1
        else:
            window = int(np.ceil(period**(window_size))) + 1
        small_t = period-window+1 if period-window+1 > 0 else 1
        for t in np.arange(small_t, period+1):
            for x in np.arange(1, capacity+1):
                if t == period-window+1:
                    next_less = val_deterministic[t-1, x-1]
                    next_same = val_deterministic[t-1, x]
                else: 
                    next_less = val_temp[t-1, x-1]
                    next_same = val_temp[t-1, x]

                action = next_same - next_less
                if action >= 1:
                    sol_temp[t, x] = 1
                elif action <= 0:
                    sol_temp[t, x] = 0
                else:
                    sol_temp[t, x] = action

                val_temp[t, x] = (1-sol_temp[t,x]**2)/2 + sol_temp[t,x]*next_same + (1-sol_temp[t,x])*next_less

                if t == period:
                    value[t, x] = val_temp[t, x]
                    solution[t, x] = sol_temp[t, x]
                    value_next[t, x] = val_temp[t-1, x]
                
                if t == period-window+1:
                    value_next[t, x] = val_deterministic[t-1, x]
    
    return value, solution


def dynamic_continuous_evaluate_solution(T, capacity, sol):
    
    val = np.zeros((T+1, capacity+1))

    for t in tqdm(np.arange(1, T+1)):
        for x in np.arange(1, capacity+1):
            next_less = val[t-1, x-1]
            next_same = val[t-1, x]

            val[t, x] = (1-sol[t,x]**2)/2 + sol[t,x]*next_same + (1-sol[t,x])*next_less

    return val

##################
#Experiments
##################

T=5000
capacity=5100
val_dynamic, sol_dynamic = dynamic_continuous_msecretary(T, capacity)
val_deterministic, sol_deterministic = deterministic_continuous_msecretary_array(T, capacity)

val_lookahead_dic = {}
sol_lookahead_dic = {}

with open('C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Data/uniform_types/val_lookahead.pkl', 'rb') as pickle_file:
    val_lookahead_dic = pickle.load(pickle_file)
with open('C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Data/uniform_types/sol_lookahead.pkl', 'rb') as pickle_file:
    sol_lookahead_dic = pickle.load(pickle_file)

val_eval_lookahead_dic = {}
windows = [1, 2, 10, 'log', 0.5, 2/3]
for window_size in windows:
    #(val_lookahead_dic[window_size], 
    #sol_lookahead_dic[window_size]) = dynamic_continuous_msecretary_lookahead(
    #    T, capacity, val_deterministic, window_size)
    val_eval_lookahead_dic[window_size] = dynamic_continuous_evaluate_solution(
        T, capacity, sol_lookahead_dic[window_size])
'''
with open('C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Data/uniform_types/val_lookahead.pkl', 'wb') as pickle_file:
    pickle.dump(val_lookahead_dic, pickle_file)
with open('C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Data/uniform_types/sol_lookahead.pkl', 'wb') as pickle_file:
    pickle.dump(sol_lookahead_dic, pickle_file)
'''


##########################################################
#Difference between lookaheads growth
#########################################################
diff_fluid_optimal = np.zeros((T+1))
diff_fluid_1 = np.zeros((T+1))
diff_1_2 = np.zeros((T+1))
diff_1_10 = np.zeros((T+1))
diff_1_log = np.zeros((T+1))
treshold_diff_fluid_1 = np.zeros((T+1))
treshold_diff_1_2 = np.zeros((T+1))

for t in np.arange(2, T):
    diff_fluid_optimal[t] =np.max(val_deterministic[t] - val_dynamic[t])
    diff_fluid_1[t] = np.max(val_deterministic[t] - val_lookahead_dic[2][t])*t
    diff_1_2[t] = np.max(val_lookahead_dic[1][t] - val_lookahead_dic[2][t])*t
    diff_1_10[t] = np.max(val_lookahead_dic[1][t] - val_lookahead_dic[10][t])*t
    diff_1_log[t] = np.max(val_lookahead_dic[1][t] - val_lookahead_dic['log'][t])*t
    treshold_diff_fluid_1[t] = np.max(sol_deterministic[t] - sol_lookahead_dic[2][t])*t
    treshold_diff_1_2[t] = np.max(sol_lookahead_dic[1][t] - sol_lookahead_dic[2][t])*t

line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(diff_fluid_optimal[:-1])
plt.plot(treshold_diff_fluid_1[:-1], label = '(Fluid - 1-Lookahead)/(1/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(treshold_diff_1_2[:-1], label = '(1-lookahead - 2-lookahead)/(1/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Difference tresholds/(1/t)')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "upper right")
plt.show()

line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(diff_fluid_optimal[:-1])
plt.plot(treshold_diff_fluid_1[:-1], label = '(Fluid - 1-Lookahead)/(1/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(treshold_diff_1_2[:-1], label = '(1-lookahead - 2-lookahead)/(1/t^2)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Difference tresholds/(corresponding rate)')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "upper right")
plt.show()

line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(diff_fluid_optimal[:-1])
plt.plot(diff_fluid_1[:-1], label = '(Fluid - 1-Lookahead)/(1og(t)/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(diff_1_2[:-1], label = '(1-lookahead - 2-lookahead)/(log(t)/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
#plt.plot(diff_1_10[:-1], label = '(1-lookahead - 10-lookahead)/(1/t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
#plt.plot(diff_1_log[:-1], label = '(1-lookahead - log-lookahead)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Difference Value Functions/(1/t)')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "upper right")
plt.show()


###############################################
#Policy path 
###############################################

T=1000
capacity=200
np.random.seed(101)
policy_optimal = np.zeros((T))
policy_lookahead = np.zeros((T))
policy_log_lookahead = np.zeros((T))
policy_fluid = np.zeros((T))
x = np.zeros((T+1, 4))
x[0,:] = capacity
arrival_types = np.random.uniform(0, 1, T+1)
for ell in np.arange(0, T):
    t = T-ell
    policy_optimal[ell] = sol_dynamic[t, int(x[ell,0])]
    policy_lookahead[ell] = sol_lookahead_dic[1][t, int(x[ell,1])]
    policy_log_lookahead[ell] = sol_lookahead_dic['log'][t, int(x[ell,2])]
    policy_fluid[ell] = 1-x[ell,3]/(t)
    if arrival_types[ell] >= policy_optimal[ell]:
        x[ell+1,0] = x[ell,0]-1
    else:
        x[ell+1,0] = x[ell,0]
    if arrival_types[ell] >= policy_lookahead[ell]:
        x[ell+1,1] = x[ell,1]-1
    else:
        x[ell+1,1] = x[ell,1]
    if arrival_types[ell] >= policy_log_lookahead[ell]:
        x[ell+1,2] = x[ell,2]-1
    else:
        x[ell+1,2] = x[ell,2]
    if arrival_types[ell] >= policy_fluid[ell]:
        x[ell+1,3] = x[ell,3]-1
    else:
        x[ell+1,3] = x[ell,3]


plt.figure(figsize=(16,10), dpi= 80)
plt.plot(policy_optimal[1:-1], label = 'Optimal', linestyle = '-', color = 'black', alpha = 0.8)
plt.plot(policy_lookahead[1:-1], label = '1-lookahead', linestyle = '-', color = 'red', alpha = 0.8)
#plt.plot(policy_log_lookahead[1:], label = 'log-lookahead', linestyle = '-.', color = 'black', alpha = 0.8)
#plt.plot(policy_fluid[1:-1], label = 'Fluid', linestyle = ':', color = 'tab:red', alpha = 0.8)
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower left")
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(x[1:,0], label = 'Optimal', linestyle = '-', color = 'black', alpha = 0.8)
plt.plot(x[1:,1], label = '1-lookahead', linestyle = '--', color = 'black', alpha = 0.8)
#plt.plot(x[1:,2], label = 'log-lookahead', linestyle = '-.', color = 'black', alpha = 0.8)
plt.plot(x[1:,3], label = 'Fluid', linestyle = ':', color = 'black', alpha = 0.8)
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower left")
plt.show()

##########################################
# Plotting
##########################################
windows = [1, 2, 'log', 0.5, 2/3]
sub_opt_gap = {window_size: np.zeros((T+1)) for window_size in windows}
sub_opt_gap_ft = {window_size: np.zeros((T+1)) for window_size in windows}
worst_state_gap = {window_size: np.zeros((T+1)) for window_size in windows}
fluid_opt_gap = np.zeros((T+1))
fluid_opt_gap_ft = np.zeros((T+1))
worst_state = np.zeros((T+1))

for t in range(2,T+1):
    for index, window_size in enumerate(windows):
        sub_opt_gap[window_size][t] = np.max(val_dynamic[t,:]-val_eval_lookahead_dic[window_size][t, :])
        sub_opt_gap_ft[window_size][t] = np.max(val_dynamic[t,:]-val_eval_lookahead_dic[window_size][t, :])/np.log(t)
        worst_state_gap[window_size][t] = np.argmax(val_dynamic[t,:]-val_eval_lookahead_dic[window_size][t, :])
    fluid_opt_gap[t] = np.max(val_deterministic[t,:]-val_dynamic[t,:]) 
    fluid_opt_gap_ft[t] = np.max(val_deterministic[t,:]-val_dynamic[t,:])/np.log(t)
    worst_state[t] = np.argmax(val_deterministic[t,:]-val_dynamic[t,:])

log_t = np.log(np.arange(1, T+1))/50

########################################################
##### Growth comparision
########################################################
line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sub_opt_gap_ft[1][11:], label = '1-lookahead Gap/log(t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(fluid_opt_gap_ft[11:], label = 'Fluid vs Optimal/log(t)', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Growth comparision')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower left")
plt.show()

line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sub_opt_gap[1], label = '1-lookahead Gap', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(fluid_opt_gap, label = 'Fluid vs Optimal', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Growth comparision')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

#######################################################
#### Worst state comparision
#######################################################
line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(worst_state_gap[1], label = 'Worst state 1-lookahead Gap', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.plot(worst_state[1:], label = 'Worst state Fluid vs Optimal', linestyle = next(line_styles), color = 'black', alpha = 0.8)
plt.title('Worst state')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower left")
plt.show()

##########################################################
#######suboptimality gap comparision for different L sizes
# ########################################################## 
line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
for window_size in windows:
    style = next(line_styles)
    color = next(colors)
    plt.plot(sub_opt_gap[window_size], label = str(window_size), linestyle = style, color = color, alpha = 0.8)
plt.title('Suboptimality gap of lookahead for different L sizes')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

line_styles = itertools.cycle(['-', '--', '-.', ':'])
colors = itertools.cycle(['black', 'red', 'blue'])
plt.figure(figsize=(16,10), dpi= 80)
for window_size in windows:
    style = next(line_styles)
    color = next(colors)
    plt.plot(sub_opt_gap_ft[window_size], label = str(window_size), linestyle = style, color = color, alpha = 0.8)
plt.title('Suboptimality gap of lookahead for different L sizes')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T ', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

#############################################
######## Value function for different L sizes
#############################################
line_styles = itertools.cycle(['--', '-.', ':'])
colors = itertools.cycle(['black', 'red'])
fix_t = 100
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(val_dynamic[fix_t,:fix_t+10], color = 'black', label='Optimal', linestyle = '-', alpha = 0.8)
plt.plot(val_eval_lookahead_dic[1][fix_t, :fix_t+10], color = 'red', label='1-lookahead', linestyle = '-', alpha = 0.8)
plt.plot(val_deterministic[fix_t,:fix_t+100], color = 'tab:red', label='Fluid', linestyle = '-', linewidth = 0.5, alpha = 0.8)
#for window_size in windows:
#    style = next(line_styles)
#    color = next(colors)
#    plt.plot(val_lookahead_dic[window_size][fix_t, :fix_t+10], color = 'tab:blue', label = str(window_size), linestyle = style, alpha = 0.8)
# Decoration
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Value function smoothing for remaining periods t = '+str(fix_t), fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('x (capacity)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

line_styles = itertools.cycle(['--', '-.', ':'])
colors = itertools.cycle(['black', 'red'])
fix_t = 5000
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot(val_dynamic[fix_t,:fix_t+10]-val_eval_lookahead_dic[1][fix_t, :fix_t+10], color = 'black', label='Optimal - 1-lookahead', linestyle = '-', alpha = 0.8)
plt.plot(val_deterministic[fix_t,:fix_t+10]-val_lookahead_dic[1][fix_t, :fix_t+10], color = 'blue', label='FLuid - 1-lookahead', linestyle = '-', alpha = 0.8)
plt.plot(val_deterministic[fix_t,:fix_t+10]-val_lookahead_dic[2][fix_t, :fix_t+10], color = 'black', label='FLuid - 2-lookahead', linestyle = '-.', alpha = 0.8)
plt.plot(val_lookahead_dic[1][fix_t,:fix_t+10]-val_lookahead_dic[2][fix_t, :fix_t+10], color = 'red', label='1-Lookahead - 2-lookahead', linestyle = '-.', alpha = 0.8)
# Decoration
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Value function smoothing for remaining periods t = '+str(fix_t), fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('x (capacity)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()


markers = itertools.cycle(['o','^','+', '*', '1', 'X'])
fix_t = 5000
plt.figure(figsize=(16,10), dpi= 80)
#for window_size in windows:
#    marker = next(markers)
#    plt.plot(val_deterministic[fix_t,:fix_t+10]-val_lookahead_dic[window_size][fix_t, :fix_t+10],
#            color = 'black', label = str(window_size), marker = marker, markersize = 5, markerfacecolor= 'none', alpha = 0.8, linewidth = 0.5)
plt.plot(val_deterministic[fix_t,:fix_t+100]-val_dynamic[fix_t,:fix_t+100], color = 'black', label='Fluid - Optimal', linestyle = '-', alpha = 0.8)
# Decoration
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.title('Difference between value functions', fontsize=20)
plt.grid(axis='both', alpha=.3)
plt.xlabel('x (capacity)', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

###########################
######## Action map
###########################
fix_t = 1000
df = pd.DataFrame(sol_lookahead_dic[.5][1:fix_t, 1:fix_t+10])
df_reversed_rows = df.iloc[::-1, :]
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed_rows, cmap='viridis',  cbar_kws={'label': 'Threshold'}, linewidths=0.5, alpha=0.6)
plt.xlabel('Remaining Capacity')
plt.ylabel('Remaining Time')
plt.title('Action Map for Optimal Solution')
plt.show()

