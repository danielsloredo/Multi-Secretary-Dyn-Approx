import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import uniform
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
    capacity = 100 #capacity upper bound
    T = 100 #Total time of observation
    probabilities = np.array([.25, .25, .25, .25]) 
    rewards = np.array([.5, 1, 1.5, 2])

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    windows = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
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
    val_deterministic = ms.deterministic_msecretary_array(T, capacity, probabilities, rewards, n_types)
    #result_lookahead[1], val_lookahead[1], sol_lookahead[1], sol_index_lookahead[1] = ms.approx_dynamic_solution(T, capacity, val_deterministic, prob_choice, rewards, vectors)
    #result_eval_lookahead[1], val_eval_lookahead[1] = ms.evaluate_solution(T, capacity, sol_index_lookahead[1], prob_choice, rewards)
    
    for window in tqdm(windows):
        result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
        result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
        suboptimality_gap[window] = np.divide(val_dynamic-val_eval_lookahead[window], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
        max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
        max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
        which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)
    
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################
    #Subotimality gap plots 
    windows_plot = [1, 10, 50]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/suboptimality_gap'
    
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

    #########################################################################################################################
    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/value_functions/1_lookahead'
    if not os.path.exists(path):
        os.makedirs(path)
    
    t_periods = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_deterministic[fix_t], color = 'tab:red', label='LP Upper Bound')
        plt.plot(val_eval_lookahead[1][fix_t], color = 'tab:blue', linestyle= '-', marker = '.', label = 'Value function using '+str(1)+'-lookahead bellman approximation')

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

    
    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/value_functions/5_lookahead'
    if not os.path.exists(path):
        os.makedirs(path)

    window = 5

    for dix, fix_t in enumerate(t_periods):
        plt.figure(figsize=(16,10), dpi= 80)
        plt.plot(val_dynamic[fix_t], color = 'black', label='Optimal value function', linestyle = '--')
        plt.plot(val_deterministic[fix_t], color = 'tab:red', label='LP Upper Bound')
        plt.plot(val_eval_lookahead[window][fix_t], color = 'tab:blue', linestyle= '-', marker = '.', label = 'Value function using '+str(5)+'-lookahead bellman approximation')

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

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/action_map/optimal'
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

        path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/action_map/lookahead'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(win)+'.png')
        plt.close()
 
    ############################################################################################################
    periods_plot = [0, 25, 50, 75]
    
    for t in periods_plot:
        df = pd.DataFrame(sol_dynamic[100-t], columns=['$r_4=.5$', '$r_3=1$', '$r_2=1.5$', '$r_1=2$'])
        # Plotting the heatmap
        plt.figure(figsize=(16,10), dpi= 80)
        sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
        plt.xlabel('Remaining Capacity')
        plt.ylabel('Reward type')
        plt.title('Action Map on T='+str(t)+' for Optimal Solution')

        min_patch = mpatches.Patch(color='blue', label='Not selected')
        max_patch = mpatches.Patch(color='red', label='Selected')
        plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

        path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/action_map/optimal'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(t)+'.png')
        plt.close()
        
        for dix, win in enumerate(windows):   
            df = pd.DataFrame(sol_lookahead[win][100-t], columns=['$r_4=.5$', '$r_3=1$', '$r_2=1.5$', '$r_1=2$'])

            # Plotting the heatmap
            plt.figure(figsize=(16,10), dpi= 80)
            sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
            plt.xlabel('Remaining Capacity')
            plt.ylabel('Reward type')
            plt.title('Action Map on T='+str(t)+' for n-lookahead='+str(win))

            min_patch = mpatches.Patch(color='blue', label='Not selected')
            max_patch = mpatches.Patch(color='red', label='Selected')
            plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

            path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/4_types/action_map/'+'period_'+str(t)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+'/action_map_'+str(win)+'.png')
            plt.close()
            
    '''
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    #For T=100 which window has below 0.0005 suboptimality gap
    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/suboptimality_gap/increasing_t'#+str(probabilities)+str(rewards)
    
    if not os.path.exists(path):
        os.makedirs(path)
    #For T=100 which window has below 0.0005 suboptimality gap
    plt.figure(figsize=(16,10), dpi= 80)
    # Sort the dictionary by keys
    sorted_data = dict(sorted(max_suboptimality_gap.items()))
    # Extract keys and values
    x = list(sorted_data.keys())
    y = list(sorted_data.values())
    # Create the line plot
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
    plt.axhline(y=.0005, color='black', linestyle='--', linewidth=2)
    plt.text(1, .0006, '0.0005', horizontalalignment='center', color='black')
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
    
    plt.savefig(path+'/maximum_sub_gap_100_p.png')
    plt.savefig(path+'/maximum_sub_gap_100_p.pdf')
    plt.close()

    ############################################################################################################
    # For T=150 which window has below 0.0005 suboptimality gap

    ######## This are global variables
    capacity = 150
    T = 150 
    windows = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
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
     
    for window in tqdm(windows):
        result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
        result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
        suboptimality_gap[window] = np.divide(val_dynamic-val_eval_lookahead[window], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
        max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
        max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
        which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)
    
    plt.figure(figsize=(16,10), dpi= 80)
    # Sort the dictionary by keys
    sorted_data = dict(sorted(max_suboptimality_gap.items()))
    # Extract keys and values
    x = list(sorted_data.keys())
    y = list(sorted_data.values())
    # Create the line plot
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
    plt.axhline(y=.0005, color='black', linestyle='--', linewidth=2)
    plt.text(1, .0006, '0.0005', horizontalalignment='center', color='black')
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
    
    plt.savefig(path+'/maximum_sub_gap_150_p.png')
    plt.savefig(path+'/maximum_sub_gap_150_p.pdf')
    plt.close()


    ############################################################################################################
    # For T=200 which window has below 0.0005 suboptimality gap

    ######## This are global variables
    capacity = 200
    T = 200 
    windows = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]

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
    
    for window in tqdm(windows):
        result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, prob_choice, rewards, vectors)
        result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
        suboptimality_gap[window] = np.divide(val_dynamic-val_eval_lookahead[window], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
        max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
        max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
        which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)
    
    plt.figure(figsize=(16,10), dpi= 80)
    # Sort the dictionary by keys
    sorted_data = dict(sorted(max_suboptimality_gap.items()))
    # Extract keys and values
    x = list(sorted_data.keys())
    y = list(sorted_data.values())
    # Create the line plot
    plt.plot(x, y, color='tab:red', marker='o', markersize=5, markerfacecolor='None', markerfacecoloralt='None', markeredgecolor='tab:red')
    plt.axhline(y=.0005, color='black', linestyle='--', linewidth=2)
    plt.text(1, .0006, '0.0005', horizontalalignment='center', color='black')
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
    
    plt.savefig(path+'/maximum_sub_gap_200_p.png')
    plt.savefig(path+'/maximum_sub_gap_200_p.pdf')
    plt.close()
    '''