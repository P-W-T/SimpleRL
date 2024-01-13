#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import itertools
import copy
import csv
import pickle
from timeit import default_timer as timer

def generate_combinations(param_grid):
    # Extract the keys and the corresponding lists of possible values
    keys, values = zip(*param_grid.items())
    
    # Use itertools.product to generate all combinations
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return combinations


def gridsearch(model_policy_fn, model_Vn_fn, model_policy_kwargs, model_Vn_kwargs, param_grid, other_args, train_fn, score_fn, score_summary_fn=np.mean, repeats=1, verbose=0):
    if repeats < 1:
        raise AssertionError('At least one run needed')
    
    best_policy = None
    best_Vn = None
    best_result = None
    best_score = -1*np.inf
    best_param = None
    
    parameter_combinations = generate_combinations(param_grid)
    
    parameters_used = []
    scores = []
    if verbose > 0:        
        start = timer()
        print(f"Start grid search: {len(parameter_combinations)} combinations")
    for num, combination in enumerate(parameter_combinations):
        rep_policy = None
        rep_Vn = None
        rep_score = -1*np.inf   
        rep_score_list = []
        rep_result = []
        
        for rep in range(repeats):
            current_policy = model_policy_fn(**model_policy_kwargs)
            current_Vn = model_Vn_fn(**model_Vn_kwargs)
            results = train_fn(model_policy=current_policy, model_Vn=current_Vn, **combination, **other_args)
            current_score = score_fn(results)
            rep_score_list.append(current_score)
            rep_result.append(results)
            
            if current_score > rep_score:
                rep_score = current_score
                rep_policy = copy.deepcopy(current_policy.state_dict())
                rep_Vn = copy.deepcopy(current_Vn.state_dict())
        
        summary_score = score_summary_fn(rep_score_list)
        scores.append(rep_score_list)
        parameters_used.append(combination)
        
        if summary_score > best_score:
            best_score = summary_score
            best_policy = rep_policy
            best_Vn = rep_Vn
            best_result = rep_result
            best_param = combination
        
        if verbose > 0:        
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60
            print(f"Grid search: {summary_score} - {num+1}/{len(parameter_combinations)} combinations - {minutes}m {seconds:2f}s")
            start = end
        
    return best_policy, best_Vn, best_result, best_param, best_score, parameters_used, scores


def dict_to_settingsfile(my_dict, filename):
    with open(filename, 'w') as file:
        file.write(f'parameter,value\n')
        # Iterate through the dictionary items
        for key, value in my_dict.items():
            # Write "key,value\n" to the file
            file.write(f'{key},{value}\n')


def score_fn_generator(cycles):
    def PG_score_fn(result):
        reward_sum, reward_len = result
        return np.median(np.median(reward_sum[-cycles:,:], axis=1), axis=0)
    return PG_score_fn
    

def PG_gridsearch(result_name, model_policy_fn, model_Vn_fn, model_policy_kwargs, model_Vn_kwargs, param_grid, other_args, train_fn, score_cycles, score_summary_fn=np.mean, repeats=1, verbose=0):
    score_fn = score_fn_generator(score_cycles)
    best_policy, best_Vn, best_result, best_param, best_score, parameters_used, scores = gridsearch(model_policy_fn, model_Vn_fn, model_policy_kwargs, model_Vn_kwargs, param_grid, other_args, train_fn, score_fn, score_summary_fn, repeats, verbose)
    
    torch.save(best_policy, result_name + "_policy.pt")
    torch.save(best_Vn, result_name + "_Vn.pt")        
    with open(result_name + "_policy_args.pkl", 'wb') as f:
        pickle.dump(model_policy_kwargs, f)    
    with open(result_name + "_Vn_args.pkl", 'wb') as f:
        pickle.dump(model_Vn_kwargs, f)
    
    for i in range(len(best_result)):
        reward_sum, reward_len = best_result[i]
        np.savetxt(result_name + "_" + str(i) + "_episoderewards.csv", reward_sum, delimiter=',')
        np.savetxt(result_name + "_" + str(i) + "_episodelength.csv", reward_len, delimiter=',')
        
    settings = {**other_args, **best_param}
    with open(result_name + "_settings.pkl", 'wb') as f:
        pickle.dump(settings, f)
    dict_to_settingsfile(settings, result_name + "_settings.txt")

    for par, score in zip(parameters_used, scores):
        # Convert scores to strings and concatenate them
        par['scores'] = ', '.join(map(str, score))

    # Determine the fieldnames for the CSV file
    fieldnames = list(parameters_used[0].keys())

    # Write to CSV
    with open(result_name + "_settings.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for d in parameters_used:
            writer.writerow(d)