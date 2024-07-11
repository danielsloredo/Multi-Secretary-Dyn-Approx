import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    n_types = 2
    capacity = 100 #capacity upper bound
    T = 100 #Total time of observation
    probabilities = np.array([.5, .5]) 
    rewards = np.array([1, 2])

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
    val_deterministic = ms.deterministic_msecretary_array(T, capacity, np.arange(1, T+1), probabilities, rewards, n_types)
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
    windows_plot = [1, 10, 50]

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
    plt.clf()

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
    plt.clf()


    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    #Action maps
    df = pd.DataFrame(sol_index_dynamic)
    #df_reversed_rows = df.iloc[::-1, :]
    df_reversed_cols = df.iloc[:, ::-1]
     # Plotting the heatmap
    plt.figure(figsize=(16,10), dpi= 80)
    sns.heatmap(df_reversed_cols, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
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
    plt.clf()

    for dix, win in enumerate(windows):   
        df = pd.DataFrame(sol_index_lookahead[win])
        #df_reversed_rows = df.iloc[::-1, :]
        df_reversed_cols = df.iloc[:, ::-1]
        # Plotting the heatmap
        plt.figure(figsize=(16,10), dpi= 80)
        sns.heatmap(df_reversed_cols, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
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
        plt.clf()

    #######
    #Action maps for specific period
    periods_plot = [0, 25, 50, 75]
    windows_plot = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for t in periods_plot:
        df = pd.DataFrame(sol_dynamic[100-t], columns=['$r_2=1$', '$r_1=2$'])
        # Plotting the heatmap
        plt.figure(figsize=(16,10), dpi= 80)
        sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
        plt.xlabel('Remaining Capacity')
        plt.ylabel('Reward type')
        plt.title('Action Map with Remaining Periods='+str(100-t)+' for Optimal Solution')

        import matplotlib.patches as mpatches
        min_patch = mpatches.Patch(color='blue', label='Not selected')
        max_patch = mpatches.Patch(color='red', label='Selected')
        plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

        path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/optimal'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/action_map_'+str(100-t)+'.png')
        plt.clf()
        
        for dix, win in enumerate(windows_plot):   
            df = pd.DataFrame(sol_lookahead[win][100-t], columns=['$r_2=1$', '$r_1=2$'])
            # Plotting the heatmap
            plt.figure(figsize=(16,10), dpi= 80)
            sns.heatmap(df.T, cmap='bwr', cbar=False, annot=False, linewidths=0.5, alpha=0.6)
            plt.xlabel('Remaining Capacity')
            plt.ylabel('Reward type')
            plt.title('Action Map with Remaining Periods='+str(100-t)+' for lookahead='+str(win))

            #vertical_lines = np.array([.25-.25/2, .5-.25/2, .75-.25/2, 1-.25/2])*(100-t)  # Change these indices as needed
            #for line in vertical_lines:
            #    plt.axvline(x=line, color='black', linestyle='--', linewidth=2)
            import matplotlib.patches as mpatches
            min_patch = mpatches.Patch(color='blue', label='Not selected')
            max_patch = mpatches.Patch(color='red', label='Selected')
            plt.legend(handles=[min_patch, max_patch], loc='upper right', bbox_to_anchor=(1.12, 1))

            path = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/Figures/2_types/action_map/'+'remaining_period_'+str(100-t)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+'/action_map_'+str(win)+'.png')
            plt.clf()
    
    