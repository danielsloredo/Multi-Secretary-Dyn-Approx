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
    n_types = 2
    capacity = 100 #capacity upper bound
    T = 100 #Total time of observation
    probabilities = np.array([.5, .5]) 
    rewards = np.array([1, 2])

    vectors = ms.generate_vectors(n_types)
    prob_choice = vectors * probabilities #p_i * u_i where u_i are binary variables.
    windows = [i for i in np.arange(5, T+5, 5)]
    windows.insert(0, 1)
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
    
    for window in tqdm(windows):
        result_lookahead[window], val_lookahead[window], sol_lookahead[window], sol_index_lookahead[window] = ms.approx_n_lookahead(T, capacity, val_deterministic, window, probabilities, rewards, vectors)
        result_eval_lookahead[window], val_eval_lookahead[window] = ms.evaluate_solution(T, capacity, sol_index_lookahead[window], prob_choice, rewards)
        suboptimality_gap[window] = np.divide(val_dynamic-val_eval_lookahead[window], val_dynamic, out=np.zeros_like(val_dynamic), where=val_dynamic != 0)
        max_suboptimality_gap[window] = np.max(suboptimality_gap[window][T])
        max_suboptimality_gap_t[window] = np.max(suboptimality_gap[window])
        which_t_max[window], which_x_max[window] = np.unravel_index(np.argmax(suboptimality_gap[window]), suboptimality_gap[window].shape)
    
    #########################################################################################################################
    windows_plot = [1, 5, 10, 20, 40]

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/suboptimality_gap'
    
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

    #######

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
    cmap_dict = {0: 'blue', 1: 'lavender', 2: 'red'}
    cmap = ListedColormap([cmap_dict[i] for i in range(3)])
    df = pd.DataFrame(sol_index_dynamic)
    #df_reversed_rows = df.iloc[::-1, :]
    df_reversed_cols = df.iloc[:, ::-1]
     # Plotting the heatmap

    plt.figure(figsize=(16,10), dpi= 80)
    sns.heatmap(df_reversed_cols, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
    plt.xlabel('Remaining Capacity')
    plt.ylabel('Remaining Time')
    plt.title('Action Map for Optimal Solution')

    min_patch = mpatches.Patch(color='blue', label='None')
    max_patch = mpatches.Patch(color='red', label='Both')
    middle_patch = mpatches.Patch(color='lavender', label='Highest Type')
    plt.legend(handles=[min_patch, middle_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

    path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/optimal'
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

        min_patch = mpatches.Patch(color='blue', label='None')
        max_patch = mpatches.Patch(color='red', label='Both')
        middle_patch = mpatches.Patch(color='lavender', label='Highest Type')
        plt.legend(handles=[min_patch, middle_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

        path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/lookahead'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(win)+'.png')
        plt.close()

    #######
    #Action maps for specific period
    periods_plot = [i for i in np.arange(5, T, 5)]
    periods_plot.insert(0, 0)
    
    for t in periods_plot:
        df = pd.DataFrame(sol_dynamic[T-t], columns=['$r_2=1$', '$r_1=2$'])
        # Plotting the heatmap
        plt.figure(figsize=(16,10), dpi= 80)
        sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
        plt.xlabel('Remaining Capacity')
        plt.ylabel('Reward type')
        plt.title('Action Map with Remaining Periods='+str(100-t)+' for Optimal Solution')

        min_patch = mpatches.Patch(color='blue', label='Not selected')
        max_patch = mpatches.Patch(color='red', label='Selected')
        plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

        path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/optimal'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(T-t)+'.png')
        plt.close()
        
        for dix, win in enumerate(windows):   
            df = pd.DataFrame(sol_lookahead[win][T-t], columns=['$r_2=1$', '$r_1=2$'])
            # Plotting the heatmap
            plt.figure(figsize=(16,10), dpi= 80)
            sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
            plt.xlabel('Remaining Capacity')
            plt.ylabel('Reward type')
            plt.title('Action Map with Remaining Periods='+str(T-t)+' for lookahead='+str(win))

            min_patch = mpatches.Patch(color='blue', label='Not selected')
            max_patch = mpatches.Patch(color='red', label='Selected')
            plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

            path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/'+'remaining_period_'+str(T-t)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+'/action_map_'+str(win)+'.png')
            plt.close()
    
    