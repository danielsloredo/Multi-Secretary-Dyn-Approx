import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os 
import cvxpy as cp
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import pickle

#############################################################
#####scheduling single server 
############################################################

def fluid_queue(probabilities, costs, T, x):
    total_cost = 0
    for t in range(0, T+1):
        x_1_t = np.maximum(0,x[0] - probabilities[0]*t)
        if t<= x[0]/probabilities[0]:
            x_2_t = x[1]
        else:
            x_2_t = np.maximum(0,x[1] - probabilities[1]*(t - x[0]/probabilities[0]))
        x_t = np.array([x_1_t, x_2_t])
        cost = np.dot(costs, x_t)
        total_cost += cost
    return total_cost

def fluid_queue_array(probabilities, costs, T, X):
    val_fluid = np.zeros((T+1, X+1, X+1))
    for t in tqdm(range(0, T+1)):
        for x1 in range(0, X+1):
            for x2 in range(0, X+1):
                cost = fluid_queue(probabilities, costs, t, np.array([x1, x2]))
                val_fluid[t][x1][x2] = cost
    return val_fluid

def dynamic_queue(T, X, probabilities, costs):
    val = np.zeros((T+1, X+1, X+1))
    sol = np.zeros((T+1, X+1, X+1), dtype=int)
    
    for t in tqdm(np.arange(0, T+1)):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    val[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                elif x1 == 0:
                    if x2==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_2 = val[t-1][x1][x2-1]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[1]*x2
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                elif x2 == 0:
                    if x1==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_1 = val[t-1][x1-1][x2]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[0]*x1
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:   
                    next_less_1 = val[t-1][x1-1][x2]
                    next_less_2 = val[t-1][x1][x2-1]
                    next_same = val[t-1][x1][x2]
                    current_cost = np.dot(costs, np.array([x1, x2]))

                    logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same <= current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                    if logic_test:
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                        sol[t][x1][x2] = 1
                    else:
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                        sol[t][x1][x2] = 0

    return val, sol

def dynamic_lookahead(T, X, val_deterministic, window_size, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    solution = np.zeros((T+1, X+1, X+1), dtype=int)

    for period in tqdm(range(0, T+1)): 
        val_temp = np.zeros((T+1, X+1, X+1))
        sol_temp = np.zeros((T+1, X+1, X+1), dtype=int)
        window = int(np.ceil(period**(window_size))) + 1
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                val_temp[0][x1][x2] = np.dot(costs, np.array([x1, x2]))
        small_t = period-window+1 if period-window > 0 else 1
        for t in np.arange(small_t, period+1):
            for x1 in np.arange(0, X+1):
                for x2 in np.arange(0, X+1):
                    if t == 0:
                        val_temp[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                    elif x1 == 0:
                        if x2==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_2 = val_deterministic[t-1][x1][x2-1]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_2 = val_temp[t-1][x1][x2-1]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[1]*x2
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            #sol_temp[t][x1][x2] = 0
                    elif x2 == 0:
                        if x1==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_1 = val_deterministic[t-1][x1-1][x2]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_1 = val_temp[t-1][x1-1][x2]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[0]*x1
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            #sol_temp[t][x1][x2] = 1
                    else:
                        if t == period-window+1:
                            next_less_1 = val_deterministic[t-1][x1-1][x2]
                            next_less_2 = val_deterministic[t-1][x1][x2-1]
                            next_same = val_deterministic[t-1][x1][x2]
                        else: 
                            next_less_1 = val_temp[t-1][x1-1][x2]
                            next_less_2 = val_temp[t-1][x1][x2-1]
                            next_same = val_temp[t-1][x1][x2]
                        
                        current_cost = np.dot(costs, np.array([x1, x2]))

                        logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same < current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                        
                        if logic_test:
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                        else:
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol_temp[t][x1][x2] = 0

                    if t == period:
                        value[t][x1][x2] = val_temp[t][x1][x2]
                        solution[t][x1][x2] = sol_temp[t][x1][x2]

    return value, solution

def dynamic_evaluate_solution(T, X, sol, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    
    for t in np.arange(0, T+1):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    value[t][x1][x2] = np.dot(costs, np.array([x1, x2]))
                elif x1 == 0:
                    if x2==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_2 = value[t-1][x1][x2-1]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[1]*x2
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            
                elif x2 == 0:
                    if x1==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_1 = value[t-1][x1-1][x2]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[0]*x1
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:        
                    next_less_1 = value[t-1][x1-1][x2]
                    next_less_2 = value[t-1][x1][x2-1]
                    next_same = value[t-1][x1][x2]
                    current_cost = np.dot(costs, np.array([x1, x2]))
                    if sol[t][x1][x2] == 1:
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                    else:
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same        

    return value

####################################################
### quadratic

def fluid_queue_quadratic(probabilities, costs, T, x):
    c1, c2 = costs[0], costs[1]  
    X1 = x[0]  
    X2 = x[1]
    p1, p2 = probabilities[0], probabilities[1] 
    u = cp.Variable(T)  
    cum_sum_u = cp.cumsum(u) * p1
    cum_sum_1_minus_u = cp.cumsum(1 - u) * p2
    z1 = cp.maximum(0, X1 - cum_sum_u)
    z2 = cp.maximum(0, X2 - cum_sum_1_minus_u)
    objective = cp.Minimize(c1 * cp.sum(z1**2) + c2 * cp.sum(z2**2))
    constraints = [u <= 1, u >= 0] 
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    #print("Status:", problem.status)
    #print("Objective value:", problem.value)
    #print("Optimal u(s):", u.value)
    total_cost = problem.value + np.dot(costs, x**2)
    return total_cost

def fluid_queue_array_quadratic(probabilities, costs, T, X, all = True):
    val_fluid = np.zeros((T+1, X+1, X+1))
    if all:
        initial = 1
    else:
        initial = T
    for t in tqdm(range(initial, T+1)):
        for x1 in tqdm(range(0, X+1)):
            for x2 in tqdm(range(0, X+1)):
                if x1 == 0 and x2 == 0:
                    cost = 0
                else:
                  cost = fluid_queue_quadratic(probabilities, costs, t, np.array([x1, x2]))
                  val_fluid[t][x1][x2] = cost
    return val_fluid

def dynamic_queue_quadratic(T, X, probabilities, costs):
    val = np.zeros((T+1, X+1, X+1))
    sol = np.zeros((T+1, X+1, X+1), dtype=int)
    
    for t in tqdm(np.arange(0, T+1)):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    val[t][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]), 2))
                elif x1 == 0:
                    if x2==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_2 = val[t-1][x1][x2-1]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[1]*x2**2
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                elif x2 == 0:
                    if x1==0:
                        val[t][x1][x2] = 0
                    else:
                        next_less_1 = val[t-1][x1-1][x2]
                        next_same = val[t-1][x1][x2]
                        current_cost = costs[0]*x1**2
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                        sol[t][x1][x2] = 1
                else:   
                    next_less_1 = val[t-1][x1-1][x2]
                    next_less_2 = val[t-1][x1][x2-1]
                    next_same = val[t-1][x1][x2]
                    current_cost = np.dot(costs, np.power(np.array([x1, x2]), 2))

                    logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same <= current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                    if logic_test:
                        val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                        sol[t][x1][x2] = 1
                    else:
                        val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                        sol[t][x1][x2] = 0

    return val, sol

def dynamic_lookahead_quadratic(T, X, val_deterministic, window_size, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    solution = np.zeros((T+1, X+1, X+1), dtype=int)

    for period in tqdm(range(0, T+1)): 
        val_temp = np.zeros((T+1, X+1, X+1))
        sol_temp = np.zeros((T+1, X+1, X+1), dtype=int)
        window = 2#int(np.ceil(period**(window_size))) + 1
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                val_temp[0][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
        small_t = period-window+1 if period-window > 0 else 1
        for t in np.arange(small_t, period+1):
            for x1 in np.arange(0, X+1):
                for x2 in np.arange(0, X+1):
                    if t == 0:
                        val_temp[t][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
                    elif x1 == 0:
                        if x2==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_2 = val_deterministic[t-1][x1][x2-1]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_2 = val_temp[t-1][x1][x2-1]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[1]*x2**2
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            #sol_temp[t][x1][x2] = 0
                    elif x2 == 0:
                        if x1==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_1 = val_deterministic[t-1][x1-1][x2]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_1 = val_temp[t-1][x1-1][x2]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[0]*x1**2
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                    else:
                        if t == period-window+1:
                            next_less_1 = val_deterministic[t-1][x1-1][x2]
                            next_less_2 = val_deterministic[t-1][x1][x2-1]
                            next_same = val_deterministic[t-1][x1][x2]
                        else: 
                            next_less_1 = val_temp[t-1][x1-1][x2]
                            next_less_2 = val_temp[t-1][x1][x2-1]
                            next_same = val_temp[t-1][x1][x2]
                        
                        current_cost = np.dot(costs, np.power(np.array([x1, x2]),2))

                        logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same < current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                        
                        if logic_test:
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                        else:
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol_temp[t][x1][x2] = 0

                    if t == period:
                        value[t][x1][x2] = val_temp[t][x1][x2]
                        solution[t][x1][x2] = sol_temp[t][x1][x2]

    return value, solution

def dynamic_evaluate_solution_quadratic(T, X, sol, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    
    for t in np.arange(0, T+1):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    value[t][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
                elif x1 == 0:
                    if x2==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_2 = value[t-1][x1][x2-1]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[1]*x2**2
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            
                elif x2 == 0:
                    if x1==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_1 = value[t-1][x1-1][x2]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[0]*x1**2
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:        
                    next_less_1 = value[t-1][x1-1][x2]
                    next_less_2 = value[t-1][x1][x2-1]
                    next_same = value[t-1][x1][x2]
                    current_cost = np.dot(costs, np.power(np.array([x1, x2]),2))
                    if sol[t][x1][x2] == 1:
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                    else:
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same        
    return value

####################################################
### 3 types

def fluid_queue_3types(probabilities, costs, T, x):
    c1, c2, c3 = costs[0], costs[1], costs[2]  
    X1 = x[0]  
    X2 = x[1]
    X3 = x[2]
    p1, p2, p3 = probabilities[0], probabilities[1] 
    u = cp.Variable(T)  
    cum_sum_u = cp.cumsum(u) * p1
    cum_sum_1_minus_u = cp.cumsum(1 - u) * p2
    z1 = cp.maximum(0, X1 - cum_sum_u)
    z2 = cp.maximum(0, X2 - cum_sum_1_minus_u)
    objective = cp.Minimize(c1 * cp.sum(z1**2) + c2 * cp.sum(z2**2))
    constraints = [u <= 1, u >= 0] 
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    #print("Status:", problem.status)
    #print("Objective value:", problem.value)
    #print("Optimal u(s):", u.value)
    total_cost = problem.value + np.dot(costs, x**2)
    return total_cost

def fluid_queue_array_quadratic(probabilities, costs, T, X, all = True):
    val_fluid = np.zeros((T+1, X+1, X+1))
    if all:
        initial = 1
    else:
        initial = T
    for t in tqdm(range(initial, T+1)):
        for x1 in range(0, X+1):
            for x2 in range(0, X+1):
                if x1 == 0 and x2 == 0:
                    cost = 0
                else:
                  cost = fluid_queue_quadratic(probabilities, costs, t, np.array([x1, x2]))
                  val_fluid[t][x1][x2] = cost
    return val_fluid

def dynamic_queue_3types(T, X, probabilities, costs):
    val = np.zeros((T+1, X+1, X+1, X+1))
    sol = np.zeros((T+1, X+1, X+1, X+1), dtype=int)
    
    for t in tqdm(np.arange(0, T+1)):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                for x3 in np.arange(0,X+1):
                    current_cost = np.dot(costs, np.power(np.array([x1, x2, x3]), 2))
                    if t == 0:
                        val[t][x1][x2][x3] = current_cost
                    elif x1 == 0 and x2==0 and x3==0:
                        val[t][x1][x2][x3] = 0
                    elif x1==0 and x2==0:
                        next_less_1 = val[t-1][x1][x2][x3-1]
                        next_same = val[t-1][x1][x2][x3]
                        val[t][x1][x2][x3] = current_cost + probabilities[2]*next_less_1+(1-probabilities[2])*next_same    
                        sol[t][x1][x2][x3] = 2           
                    elif x1==0 and x3==0:
                        next_less_1 = val[t-1][x1][x2-1][x3]
                        next_same = val[t-1][x1][x2][x3]
                        val[t][x1][x2][x3] = current_cost + probabilities[1]*next_less_1+(1-probabilities[1])*next_same
                        sol[t][x1][x2][x3] = 1
                    elif x2==0 and x3==0:
                        next_less_1 = val[t-1][x1-1][x2][x3]
                        next_same = val[t-1][x1][x2][x3]
                        val[t][x1][x2][x3] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                        sol[t][x1][x2][x3] = 0
                    elif x1==0:
                        next_less_1 = val[t-1][x1][x2-1][x3]
                        next_less_2 = val[t-1][x1][x2][x3-1]
                        next_same = val[t-1][x1][x2][x3]
                        #Im here
                        logic_test = (probabilities[1]*next_less_1-probabilities[1]*next_same <= probabilities[2]*next_less_2-probabilities[2]*next_same) 
                        if logic_test:
                            val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol[t][x1][x2] = 1
                        else:
                            val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol[t][x1][x2] = 0

                    elif x2==0:
                    elif x3==0:

                        else:
                            next_less_2 = val[t-1][x1][x2-1]
                            next_same = val[t-1][x1][x2]
                            current_cost = costs[1]*x2**2
                            val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                    elif x2 == 0:
                        if x1==0:
                            val[t][x1][x2] = 0
                        else:
                            next_less_1 = val[t-1][x1-1][x2]
                            next_same = val[t-1][x1][x2]
                            current_cost = costs[0]*x1**2
                            val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol[t][x1][x2] = 1
                    else:   
                        next_less_1 = val[t-1][x1-1][x2]
                        next_less_2 = val[t-1][x1][x2-1]
                        next_same = val[t-1][x1][x2]
                        current_cost = np.dot(costs, np.power(np.array([x1, x2]), 2))

                        logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same <= current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                        if logic_test:
                            val[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol[t][x1][x2] = 1
                        else:
                            val[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol[t][x1][x2] = 0

    return val, sol

def dynamic_lookahead_quadratic(T, X, val_deterministic, window_size, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    solution = np.zeros((T+1, X+1, X+1), dtype=int)

    for period in tqdm(range(0, T+1)): 
        val_temp = np.zeros((T+1, X+1, X+1))
        sol_temp = np.zeros((T+1, X+1, X+1), dtype=int)
        window = 2#int(np.ceil(period**(window_size))) + 1
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                val_temp[0][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
        small_t = period-window+1 if period-window > 0 else 1
        for t in np.arange(small_t, period+1):
            for x1 in np.arange(0, X+1):
                for x2 in np.arange(0, X+1):
                    if t == 0:
                        val_temp[t][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
                    elif x1 == 0:
                        if x2==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_2 = val_deterministic[t-1][x1][x2-1]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_2 = val_temp[t-1][x1][x2-1]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[1]*x2**2
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            #sol_temp[t][x1][x2] = 0
                    elif x2 == 0:
                        if x1==0:
                            val_temp[t][x1][x2] = 0
                        else:
                            if t == period-window+1:
                                next_less_1 = val_deterministic[t-1][x1-1][x2]
                                next_same = val_deterministic[t-1][x1][x2]
                            else:
                                next_less_1 = val_temp[t-1][x1-1][x2]
                                next_same = val_temp[t-1][x1][x2]
                            current_cost = costs[0]*x1**2
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                    else:
                        if t == period-window+1:
                            next_less_1 = val_deterministic[t-1][x1-1][x2]
                            next_less_2 = val_deterministic[t-1][x1][x2-1]
                            next_same = val_deterministic[t-1][x1][x2]
                        else: 
                            next_less_1 = val_temp[t-1][x1-1][x2]
                            next_less_2 = val_temp[t-1][x1][x2-1]
                            next_same = val_temp[t-1][x1][x2]
                        
                        current_cost = np.dot(costs, np.power(np.array([x1, x2]),2))

                        logic_test = (current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same < current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same) 
                        
                        if logic_test:
                            val_temp[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                            sol_temp[t][x1][x2] = 1
                        else:
                            val_temp[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            sol_temp[t][x1][x2] = 0

                    if t == period:
                        value[t][x1][x2] = val_temp[t][x1][x2]
                        solution[t][x1][x2] = sol_temp[t][x1][x2]

    return value, solution

def dynamic_evaluate_solution_quadratic(T, X, sol, probabilities, costs):
    value = np.zeros((T+1, X+1, X+1))
    
    for t in np.arange(0, T+1):
        for x1 in np.arange(0, X+1):
            for x2 in np.arange(0, X+1):
                if t == 0:
                    value[t][x1][x2] = np.dot(costs, np.power(np.array([x1, x2]),2))
                elif x1 == 0:
                    if x2==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_2 = value[t-1][x1][x2-1]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[1]*x2**2
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same
                            
                elif x2 == 0:
                    if x1==0:
                        value[t][x1][x2] = 0
                    else:
                        next_less_1 = value[t-1][x1-1][x2]
                        next_same = value[t-1][x1][x2]
                        current_cost = costs[0]*x1**2
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                else:        
                    next_less_1 = value[t-1][x1-1][x2]
                    next_less_2 = value[t-1][x1][x2-1]
                    next_same = value[t-1][x1][x2]
                    current_cost = np.dot(costs, np.power(np.array([x1, x2]),2))
                    if sol[t][x1][x2] == 1:
                        value[t][x1][x2] = current_cost + probabilities[0]*next_less_1+(1-probabilities[0])*next_same
                    else:
                        value[t][x1][x2] = current_cost + probabilities[1]*next_less_2+(1-probabilities[1])*next_same        
    return value



############################################################################
############################################################################
##  Main code
############################################################################
############################################################################
probabilities = np.array([0.5, 0.5])
costs = np.array([2, 1.5])
T = 100
X = 100
window_size = 4/5

val_fluid_queue = fluid_queue_array(probabilities, costs, T, X)
val_dp_queue, sol_dp_queue = dynamic_queue(T, X, probabilities, costs)
val_lookahead, sol_lookahead = dynamic_lookahead(T, X, val_fluid_queue, window_size, probabilities, costs)
val_eval_lookahead = dynamic_evaluate_solution(T, X, sol_lookahead, probabilities, costs)
sub_opt_gap = np.zeros(T+1)
for t in range(0, T+1):
    sub_opt_gap[t] = np.max(val_eval_lookahead[t]-val_dp_queue[t])


probabilities = np.array([0.5, 0.5])
costs = np.array([2, 1.5])
T = 50
X = 100
window_size = 2/3

#val_fluid_queue_2 = fluid_queue_array_quadratic(probabilities, costs, T, X, all = False)
val_dp_queue_2, sol_dp_queue_2 = dynamic_queue_quadratic(T, X, probabilities, costs)
val_lookahead_2, sol_lookahead_2 = dynamic_lookahead_quadratic(T, X, val_fluid_queue_2, window_size, probabilities, costs)
val_eval_lookahead_2 = dynamic_evaluate_solution_quadratic(T, X, sol_lookahead_2, probabilities, costs)
sub_opt_gap_2 = np.zeros(T+1)
for t in range(0, T+1):
    sub_opt_gap_2[t] = np.max(val_eval_lookahead_2[t]-val_dp_queue_2[t])
 

'''
with open('C:/Users/danie/OneDrive/Documents/multisecretary-dyn-approx/Multi-Secretary-Dyn-Approx/cmu_rule/val_fluid_queue_2.pkl', 'wb') as pickle_file:
    pickle.dump(val_fluid_queue_2, pickle_file)
'''


with open('C:/Users/danie/OneDrive/Documents/multisecretary-dyn-approx/Multi-Secretary-Dyn-Approx/cmu_rule/val_fluid_queue_2.pkl', 'rb') as pickle_file:
    val_fluid_queue_2 = pickle.load(pickle_file)

val_fluid_queue_2_100 = np.copy(val_fluid_queue_2)
############################################################################
############################################################################
#Plots for Presentation
############################################################################
############################################################################

path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'
path_presentation = path_0 + 'Figures/presentation/'
if not os.path.exists(path_presentation):
    os.makedirs(path_presentation)

#############
## Regret
##############
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sub_opt_gap, color = 'tab:blue', label='Regret', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_regret.png')
plt.close()

############
#Plot of value functions surface
############
#linear
t = 100
X = 100 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue[t, 0:X+1, 0:X+1] 
z2 = val_dp_queue[t, 0:X+1, 0:X+1]
#z3 = val_eval_lookahead[t, 0:X+1, 0:X+1]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
#surf3 = ax.plot_surface(x1, x2, z3, cmap='Grays')
# Add labels and title
ax.set_xlabel('K2')
ax.set_ylabel('K1')
ax.set_zlabel('Cost')
#ax.set_title(f'Surface Plot of value functions at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

#############
#Plot of difference between value functions 
#############
z = val_eval_lookahead[t, 0:X+1, 0:X+1]-val_fluid_queue[t, 0:X+1, 0:X+1]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='viridis')
# Add labels and title
ax.set_xlabel('K2')
ax.set_ylabel('K1')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


#################
## Plots of derivatives of value functions
#################
t=100
k1 = 30
derivative_x_2_fluid = np.diff(val_fluid_queue[t], n=1, axis=1)
derivative_x_2_dp = np.diff(val_dp_queue[t], n=1, axis=1)
derivative_x_2_lookahead = np.diff(val_lookahead[t], n=1, axis=1)
derivative_x_2_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 1)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_type2.png')
plt.close()


plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_2_lookahead[k1], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_lookahead_type2.png')
plt.close()

k2 = 30
derivative_x_1_fluid = np.diff(val_fluid_queue[t], n=1, axis=0)
derivative_x_1_dp = np.diff(val_dp_queue[t], n=1, axis=0)
derivative_x_1_lookahead = np.diff(val_lookahead[t], n=1, axis=0)
derivative_x_1_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 0)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_type1.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_1_lookahead[:,k2], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.savefig(path_presentation+'queue_derivative_lookahead_type1.png')
plt.close()


t = 50
derivative_x_2 = np.diff(val_dp_queue[t], n=1,axis=1)
derivative_x_2_fluid = np.diff(val_fluid_queue[t], n=1,axis=1)
# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_2[1:,:]  # Slice to exclude zeros for x1 and x2
z2 = derivative_x_2_fluid[1:,:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
# Add labels and title
ax.set_xlabel('x2')
ax.set_ylabel('x1')
ax.set_zlabel('Difference Cost')
ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


derivative_x_1 = np.diff(val_dp_queue[t], n=1,axis=0)
derivative_x_1_fluid = np.diff(val_fluid_queue[t], n=1,axis=0)
# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_1[:,1:]  # Slice to exclude zeros for x1 and x2
z2 = derivative_x_1_fluid[:,1:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
# Add labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Difference Cost')
ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


t=100
X=100
k1 = 30
k2 = 30

derivative_x_2_fluid = np.diff(val_fluid_queue[t], n=1, axis=1)
derivative_x_2_dp = np.diff(val_dp_queue[t], n=1, axis=1)
derivative_x_2_lookahead = np.diff(val_lookahead[t], n=1, axis=1)
derivative_x_2_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 1)

derivative_x_1_fluid = np.diff(val_fluid_queue[t], n=1, axis=0)
derivative_x_1_dp = np.diff(val_dp_queue[t], n=1, axis=0)
derivative_x_1_lookahead = np.diff(val_lookahead[t], n=1, axis=0)
derivative_x_1_eval_lookahead = np.diff(val_eval_lookahead[t], n=1, axis = 0)

diff_derivatives = np.zeros((X, X))
diff_derivatives_fluid = np.zeros((X, X))
for k1 in range(X):
    for k2 in range(X):
        diff_derivatives[k1,k2] = .5*derivative_x_1_dp[k1,k2] - .5*derivative_x_2_dp[k1,k2]
        diff_derivatives_fluid[k1,k2] = .5*derivative_x_1_fluid[k1,k2] - .5*derivative_x_2_fluid[k1,k2]

plt.plot(diff_derivatives[40,:])
plt.plot(diff_derivatives_fluid[40,:])
plt.show()
plt.plot(diff_derivatives[:,40])
plt.plot(diff_derivatives_fluid[:,40])
plt.show()



############################################################################
############################################################################
#Plots for Presentation Quadratic
############################################################################
############################################################################

path_0 = 'C:/Users/danie/Documents/Multi-Secretary-Dyn-Approx/'
path_presentation = path_0 + 'Figures/presentation/'
if not os.path.exists(path_presentation):
    os.makedirs(path_presentation)

########################################################################
## Action Map 
########################################################################
cmap_dict = {0: 'tab:blue', 1: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(2)])
df = pd.DataFrame(sol_dp_queue_2[50][:X+1,:X+1])
df_reversed = df.iloc[::-1, :]
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
plt.xlabel('Q2')
plt.ylabel('Q1')
plt.title('Optimal Solution Policy')
patch_0 = mpatches.Patch(color='tab:blue', label='Type 2')
patch_1 = mpatches.Patch(color='tab:red', label='Type 1')
plt.legend(handles=[patch_0, patch_1], loc='upper left')
plt.show()

cmap_dict = {0: 'tab:blue', 1: 'tab:red'}
cmap = ListedColormap([cmap_dict[i] for i in range(2)])
df = pd.DataFrame(sol_lookahead_2[50][:X+1,:X+1])
df_reversed = df.iloc[::-1, :]
plt.figure(figsize=(16,10), dpi= 80)
sns.heatmap(df_reversed, cmap=cmap, cbar=False, annot=False, linewidths=0.5, alpha=0.6)
plt.xlabel('Q2')
plt.ylabel('Q1')
plt.title('Lookahead Policy')
patch_0 = mpatches.Patch(color='tab:blue', label='Type 2')
patch_1 = mpatches.Patch(color='tab:red', label='Type 1')
plt.legend(handles=[patch_0, patch_1], loc='upper left')
plt.show()

#############
## Regret
##############
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sub_opt_gap_2[1:], color = 'tab:blue', label='Regret', linestyle = '-', marker = '.', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('T', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()
plt.savefig(path_presentation+'queue_regret_2.png')
plt.close()

############
#Plot of value functions surface
############
t = 100
X = 200 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue_2[t, 0:X+1, 0:X+1] 
z2 = val_dp_queue_2[t, 0:X+1, 0:X+1]
#z3 = val_eval_lookahead[t, 0:X+1, 0:X+1]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Blues')
#surf3 = ax.plot_surface(x1, x2, z3, cmap='Grays')
# Add labels and title
ax.set_xlabel('Q2')
ax.set_ylabel('Q1')
ax.set_zlabel('Cost')
#ax.set_title(f'Surface Plot of value functions at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

#############
#Plot of difference between value functions 
#############
t = 100
X = 200 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue_2[t, 0:X+1, 0:X+1] 
z2 = val_dp_queue_2[t, 0:X+1, 0:X+1]
z = val_dp_queue_2[t, 0:X+1, 0:X+1]-val_fluid_queue_2[t, 0:X+1, 0:X+1]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='viridis')
# Add labels and title
ax.set_xlabel('Q2')
ax.set_ylabel('Q1')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

t = 50
X = 100 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue_2[t, 0:X+1, 0:X+1] 
z2 = val_dp_queue_2[t, 0:X+1, 0:X+1]
z = val_dp_queue_2[t, 0:X+1, 0:X+1]-val_fluid_queue_2[t, 0:X+1, 0:X+1]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='viridis')
# Add labels and title
ax.set_xlabel('Q2')
ax.set_ylabel('Q1')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

###################
# Plot of the Gradient
###################

t = 50
X = 100 
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = val_fluid_queue_2[t, 0:X+1, 0:X+1] 
grad_1_fluid, grad_2_fluid = np.gradient(z)
z2 = val_dp_queue_2[t, 0:X+1, 0:X+1] 
grad_1_dp, grad_2_dp = np.gradient(z2)
z3 = val_lookahead_2[t, 0:X+1, 0:X+1]
grad_1_lookahead, grad_2_lookahead = np.gradient(z3)

plt.figure(figsize=(16,10), dpi= 80)
plt.contourf(x1,x2, z, cmap = 'viridis')
plt.colorbar()
plt.quiver(x1,x2,grad_1_fluid, grad_2_fluid, color = 'red')
plt.quiver(x1,x2,grad_1_dp, grad_2_dp, color = 'black')
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.show()

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, grad_1_fluid, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, grad_1_dp, cmap='Blues')
surf3 = ax.plot_surface(x1, x2, grad_1_lookahead, cmap='Grays')
# Add labels and title
ax.set_xlabel('Q2')
ax.set_ylabel('Q1')
ax.set_zlabel('Cost')
#ax.set_title(f'Surface Plot of value functions at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, grad_2_fluid, cmap='Reds')
surf2 = ax.plot_surface(x1, x2, grad_2_dp, cmap='Blues')
surf3 = ax.plot_surface(x1, x2, grad_2_lookahead, cmap='Grays')
# Add labels and title
ax.set_xlabel('Q2')
ax.set_ylabel('Q1')
ax.set_zlabel('Cost')
#ax.set_title(f'Surface Plot of value functions at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


#################
## Plots of derivatives of value functions
#################
t=100
k1 = 20
k2 = 20

derivative_x_2_fluid = np.diff(val_fluid_queue_2[t], n=1, axis=1)
derivative_x_2_dp = np.diff(val_dp_queue_2[t], n=1, axis=1)
derivative_x_2_lookahead = np.diff(val_lookahead_2[t], n=1, axis=1)
derivative_x_2_eval_lookahead = np.diff(val_eval_lookahead_2[t], n=1, axis = 1)

sec_derivative_x_2_fluid =np.diff(np.diff(val_fluid_queue_2[t], n=1, axis=1), n =1 , axis=1)
sec_derivative_x_2_dp = np.diff(np.diff(val_dp_queue_2[t], n=1, axis=1), n =1 , axis=1)
sec_derivative_x_2_lookahead = np.diff(np.diff(val_lookahead_2[t], n=1, axis=1), n =1 , axis=1)
sec_derivative_x_2_eval_lookahead = np.diff(np.diff(val_eval_lookahead_2[t], n=1, axis = 1), n =1 , axis=1)

derivative_x_1_fluid = np.diff(val_fluid_queue_2[t], n=1, axis=0)
derivative_x_1_dp = np.diff(val_dp_queue_2[t], n=1, axis=0)
derivative_x_1_lookahead = np.diff(val_lookahead_2[t], n=1, axis=0)
derivative_x_1_eval_lookahead = np.diff(val_eval_lookahead_2[t], n=1, axis = 0)

sec_derivative_x_1_fluid =np.diff(np.diff(val_fluid_queue_2[t], n=1, axis=0), n =1 , axis=0)
sec_derivative_x_1_dp = np.diff(np.diff(val_dp_queue_2[t], n=1, axis=0), n =1 , axis=0)
sec_derivative_x_1_lookahead = np.diff(np.diff(val_lookahead_2[t], n=1, axis=0), n =1 , axis=0)
sec_derivative_x_1_eval_lookahead = np.diff(np.diff(val_eval_lookahead_2[t], n=1, axis = 0), n =1 , axis=0)

t2= 100
derivative_x_2_fluid_100 = np.diff(val_fluid_queue_2_100[t2], n=1, axis=1)
sec_derivative_x_2_fluid_100 =np.diff(np.diff(val_fluid_queue_2_100[t2], n=1, axis=1), n =1 , axis=1)
derivative_x_1_fluid_100 = np.diff(val_fluid_queue_2_100[t2], n=1, axis=0)
sec_derivative_x_1_fluid_100 =np.diff(np.diff(val_fluid_queue_2_100[t2], n=1, axis=0), n =1 , axis=0)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_2_lookahead[k1], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$Q_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()
plt.savefig(path_presentation+'queue_derivative_lookahead_type2_2.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sec_derivative_x_2_fluid[k1], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(sec_derivative_x_2_dp[k1], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(sec_derivative_x_2_lookahead[k1], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$Q_2$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(derivative_x_1_lookahead[:,k2], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()
plt.savefig(path_presentation+'queue_derivative_lookahead_type1_2.png')
plt.close()

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(sec_derivative_x_1_fluid[:,k2], color = 'tab:red', label='Deterministic', linestyle = '-', marker = '', fillstyle = 'none')
plt.plot(sec_derivative_x_1_dp[:,k2], color = 'tab:blue', label='Optimal', linestyle = '-', marker = '.', fillstyle = 'none')
plt.plot(sec_derivative_x_1_lookahead[:,k2], color = 'tab:gray', label='Lookahead', linestyle = '-', marker = '', fillstyle = 'none')
plt.xticks(sec_rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)
plt.xlabel('$K_1$', fontsize = 14)
plt.gca().spines["top"].set_alpha(0.3)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.legend(loc = "lower right")
plt.show()

# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_1_fluid[:,1:] - derivative_x_1_dp[:X+1,1:X+2]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# Create meshgrid for x1 and x2
X = 99
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_2_fluid[1:,:] - derivative_x_2_dp[1:X+2,:X+1]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


# Create meshgrid for x1 and x2
X = 98
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = sec_derivative_x_1_dp[:X+1,2:X+3]  # Slice to exclude zeros for x1 and x2
z2 = sec_derivative_x_1_fluid[:,2:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Reds')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


X = 98
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = sec_derivative_x_2_dp[2:X+3,:X+1]  # Slice to exclude zeros for x1 and x2
z2 = sec_derivative_x_2_fluid[2:,:]
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
surf2 = ax.plot_surface(x1, x2, z2, cmap='Reds')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


X = 98
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = sec_derivative_x_1_fluid[:,2:] - sec_derivative_x_1_dp[:X+1,2:X+3] 
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# Create meshgrid for x1 and x2
X = 199
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = derivative_x_1_fluid_100[:,1:] - derivative_x_1_dp[:,1:]  # Slice to exclude zeros for x1 and x2
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

X = 198
x1 = np.arange(0, X + 1)
x2 = np.arange(0, X + 1)
x1, x2 = np.meshgrid(x1, x2)
z = sec_derivative_x_1_fluid_100[:,2:] - sec_derivative_x_1_dp[:,2:] 
# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='Blues')
# Add labels and title
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Difference Cost')
#ax.set_title(f'Surface Plot of val_fluid_queue at t={t}')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
