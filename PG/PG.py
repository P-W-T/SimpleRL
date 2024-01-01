#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import copy


def play_env(observation, env, model, steps, device='cpu'):
    """
    Simulate an environment for a specified number of steps using a given model.

    Parameters:
    observation (np.array): The previous observation from the environment.
    env (gym.Env): The Gym environment to be simulated.
    model (torch.nn.Module): The model that predicts the action based on the observation.
    steps (int): The number of steps to simulate the environment.
    device (str): The device ('cpu' or 'cuda') used for tensor computations.

    Returns:
    tuple: A tuple containing the environment, list of observations, list of actions,
           list of rewards, and a boolean indicating if the simulation ended because
           the environment was 'done'.
    """
    
    # Initialize lists to store observations, actions, and rewards
    observation_list = []
    action_list = []
    reward_list = []
    final = False # Flag to indicate if the environment reached a terminal state
    
    terminated, truncated = False, False

    step = 0
    while not terminated and not truncated and step < steps:
        observation_list.append(observation)
        # Generate an action from the model based on the current observation
        # Unsqueeze is used to add batch dimension, which is required for most PyTorch models
        with torch.no_grad():
            action = model.sample(torch.unsqueeze(torch.tensor(observation, dtype=torch.float32, device=device), dim=0))

        # Execute the action in the environment and get the next observation, reward, and status
        observation, reward, terminated, truncated, _ = env.step(action)
        reward_list.append(reward)
        action_list.append(action)
        step += 1
    
    # If the environment has reached a terminal state, reset it
    if terminated or truncated:
        observation, _ = env.reset()
        final = True
    
    # Add the final observation to the observation list
    observation_list.append(observation)
    
    return np.array(observation_list), action_list, reward_list, final

    
def advantage_GAE(observations, actions, rewards, model_Vn, final, discount, lam, device='cpu'):      
    """
    Calculate the Generalized Advantage Estimation (GAE) for given observations, actions, and rewards.

    Parameters:
    observations (np.array): Observations from the environment.
    actions (np.array): Actions taken in the environment.
    rewards (np.array): Rewards received from the environment.
    model_Vn (torch.nn.Module): A neural network model that estimates the value function.
    final (bool): A flag indicating if the final state is terminal.
    discount (float): Discount factor for future rewards.
    lam (float): Lambda parameter for GAE.
    device (str): The device ('cpu' or 'cuda') used for tensor computations.

    Returns:
    tuple: A tuple containing discounted rewards, processed observations,
           processed actions, and calculated advantages.
    """
    
    factor = lam*discount
    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    actions = torch.unsqueeze(torch.tensor(actions, dtype=torch.float32, device=device), dim=-1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Evaluate the value function for each observation
    with torch.no_grad():
        raw_Vn = torch.squeeze(model_Vn(observations))
    
    # Prepare value function estimates for current and next states
    Vn = raw_Vn[:-1]
    Vn1 = raw_Vn[1:]
    if final:
        Vn1[-1] = 0.0 # If final state is terminal, set next state's value to 0
    
    # Compute delta = reward + discount * V(next state) - V(current state)
    delta = rewards + discount*Vn1 - Vn
    
    # Initialize tensors for advantages and discounted rewards
    new_advantages = torch.zeros(len(delta), dtype=torch.float32, device=device) 
    new_advantages[-1] = delta[-1]
    discount_rewards = torch.zeros(len(rewards), dtype=torch.float32, device=device)
    discount_rewards[-1] = rewards[-1] + discount * Vn1[-1]
    
    # Calculate advantages and discounted rewards backwards
    for num in range(len(new_advantages)-2, -1, -1):
        new_advantages[num] = delta[num] + factor*new_advantages[num + 1]
        discount_rewards[num] = rewards[num] + discount * discount_rewards[num+1]
    
    new_observations = observations[:-1]
    new_actions = actions
    
    return discount_rewards, new_observations, new_actions, new_advantages


def loss_fn(model, observation_tensor, action_tensor, weight_tensor, beta):
    """
    Calculate the loss for a policy network with entropy regularization.

    Parameters:
    model (torch.nn.Module): The policy model that outputs action probabilities.
    observation_tensor (torch.Tensor): Tensor of observations.
    action_tensor (torch.Tensor): Tensor of actions taken.
    weight_tensor (torch.Tensor): Tensor of weights (advantages or returns).
    beta (float): Coefficient for entropy regularization.

    Returns:
    torch.Tensor: The calculated loss value.
    """
    # Compute the log probability of the actions taken
    logp = torch.squeeze(model.log_prob(observation_tensor, action_tensor))
    #print(action_tensor)
    #print(model.log_prob(observation_tensor, action_tensor))
    #print(logp)
    # Calculate the policy gradient loss
    policy_loss = -(logp * weight_tensor).mean()

    # Calculate the entropy regularization term
    entropy_regularization = beta * (torch.exp(logp) * logp).mean()

    # The total loss is the sum of policy loss and entropy regularization
    return policy_loss - entropy_regularization


def train_agent(model_policy, model_Vn, env_name, lam, discount, beta, train_Vn,
                optimizer_policy, optimizer_Vn,
                n_games_per_cycle=1, gradient_clip_policy=None,
                gradient_clip_Vn=None, median_stop_threshold=None,
                median_stop_patience=None, device='cpu', inference_device='cpu'):
    """
    Train the agent for a specified number of games per cycle.

    Parameters:
    model_policy (torch.nn.Module): The policy model.
    model_Vn (torch.nn.Module): The value network model.
    env_name (str): Name of the Gym environment.
    lam (float): Lambda parameter for GAE.
    discount (float): Discount factor for future rewards.
    beta (float): Coefficient for entropy regularization.
    train_Vn (int): Number of times to train the value network per iteration.
    optimizer_policy (torch.optim.Optimizer): Optimizer for the policy network.
    optimizer_Vn (torch.optim.Optimizer): Optimizer for the value network.
    n_games_per_cycle (int, optional): Number of games to play per cycle.
    gradient_clip_policy (float, optional): Gradient clipping value for policy network.
    gradient_clip_Vn (float, optional): Gradient clipping value for value network.
    median_stop_threshold (float, optional): Threshold for early stopping based on median reward.
    median_stop_patience (int, optional): Patience for early stopping based on median reward.
    device (str, optional): The device to run the training computations ('cpu' or 'cuda').
    inference_device (str, optional): The device to run the inference computations ('cpu' or 'cuda').
    
    Returns:
    tuple: Tuple containing arrays of sum of rewards and lengths of each game.
    """
    
    env = gym.make(env_name)
    last_observation, _ = env.reset()
    
    reward_sum = np.zeros(n_games_per_cycle)
    reward_len = np.zeros(n_games_per_cycle)
    
    discount_rewards_list = None
    observation_list = None
    action_list = None
    advantage_list = None
    
    model_policy, model_Vn = model_policy.to(inference_device), model_Vn.to(inference_device)
    
    for i in range(n_games_per_cycle):        
        observation_cur, action_cur, reward_cur, final = play_env(last_observation, env, model_policy, np.inf, inference_device)
        last_observation = observation_cur[-1]
        reward_sum[i] = np.sum(reward_cur)
        reward_len[i] = len(reward_cur)      
        
        discount_rewards, new_observations, new_actions, new_advantages = advantage_GAE(observation_cur, action_cur, reward_cur, model_Vn, final, discount, lam, inference_device)
        
        if discount_rewards_list is None:
            discount_rewards_list = discount_rewards
            observation_list = new_observations
            action_list = new_actions
            advantage_list = new_advantages
        else:
            discount_rewards_list = torch.cat((discount_rewards_list, discount_rewards), dim=0)
            observation_list = torch.cat((observation_list, new_observations), dim=0)
            action_list = torch.cat((action_list, new_actions), dim=0)
            advantage_list = torch.cat((advantage_list, new_advantages), dim=0)
    
    if inference_device != device:
        model_policy, model_Vn = model_policy.to(device), model_Vn.to(device)
        discount_rewards_list = discount_rewards_list.to(device)
        observation_list = observation_list.to(device)
        action_list = action_list.to(device)
        advantage_list = advantage_list.to(device)
    
    optimizer_policy.zero_grad()
    loss_policy = loss_fn(model_policy, observation_list, action_list, advantage_list, beta)    
    loss_policy.backward()
    if gradient_clip_policy is not None:
        nn.utils.clip_grad_norm_(model_policy.parameters(), gradient_clip_policy)
    optimizer_policy.step()
    
    for _ in range(int(train_Vn)):
        optimizer_Vn.zero_grad()
        Vn = torch.squeeze(model_Vn(observation_list))
        loss_Vn = nn.MSELoss()(Vn, discount_rewards_list)
        loss_Vn.backward()
        if gradient_clip_Vn is not None:
            nn.utils.clip_grad_norm_(model_Vn.parameters(), gradient_clip_Vn)
        optimizer_Vn.step()
    
    return reward_sum, reward_len


def train(model_policy, model_Vn, save_name, env_name, lam, discount, beta, train_Vn,
          n_cycles, n_games_per_cycle, report_updates,
          gradient_clip_policy=None, gradient_clip_Vn=None, median_stop_threshold=None,
          median_stop_patience=None, length=False, save_cycles=None, 
          policy_lr=0.001, policy_beta1=0.9, policy_beta2=0.999, policy_eps=1e-08, 
          Vn_lr=0.001, Vn_beta1=0.9, Vn_beta2=0.999, Vn_eps=1e-08, device='cpu', inference_device='cpu'):
    """
    Train the policy and value network models over multiple cycles.

    Parameters:
    model_policy (torch.nn.Module): The policy model.
    model_Vn (torch.nn.Module): The value network model.
    env_name (str): Name of the Gym environment.
    save_name (str): Base name for saving model states and the outputs.
    lam (float): Lambda parameter for GAE.
    discount (float): Discount factor for future rewards.
    beta (float): Coefficient for entropy regularization.
    train_Vn (int): Number of times to train the value network per cycle.
    n_cycles (int): Total number of training cycles.
    n_games_per_cycle (int): Number of games to play per cycle.
    report_updates (int): Frequency of reporting progress.
    optimizer_policy (torch.optim.Optimizer): Optimizer for the policy network.
    optimizer_Vn (torch.optim.Optimizer): Optimizer for the value network.
    gradient_clip_policy (float, optional): Gradient clipping value for policy network.
    gradient_clip_Vn (float, optional): Gradient clipping value for value network.
    median_stop_threshold (float, optional): Threshold for early stopping based on median reward.
    median_stop_patience (int, optional): Patience for early stopping based on median reward.
    length (bool, optional): Whether to report game lengths instead of rewards.
    save_cycles (int): Frequency of saving model states and statistics.
    policy_lr (float): Learning rate for the policy optimizer.
    policy_beta1 (float): Beta1 parameter for the policy optimizer (Adam).
    policy_beta2 (float): Beta2 parameter for the policy optimizer (Adam).
    policy_eps (float): Epsilon parameter for the policy optimizer (Adam).
    Vn_lr (float): Learning rate for the value network optimizer.
    Vn_beta1 (float): Beta1 parameter for the value network optimizer (Adam).
    Vn_beta2 (float): Beta2 parameter for the value network optimizer (Adam).
    Vn_eps (float): Epsilon parameter for the value network optimizer (Adam).  
    device (str, optional): The device to run the training computations ('cpu' or 'cuda').
    inference_device (str, optional): The device to run the inference computations ('cpu' or 'cuda').

    Returns:
    tuple: Tuple containing arrays of sum of rewards and lengths of each game.
    """
    
    reward_sum = np.zeros((n_cycles, n_games_per_cycle))
    reward_len = np.zeros((n_cycles, n_games_per_cycle))
    
    optimizer_policy = torch.optim.Adam(model_policy.parameters(), lr=policy_lr, betas=(policy_beta1, policy_beta2), eps=policy_eps)
    optimizer_Vn = torch.optim.Adam(model_Vn.parameters(), lr=Vn_lr, betas=(Vn_beta1, Vn_beta2), eps=Vn_eps)
    
    for cycle in range(n_cycles):
        curr_sum, curr_len = train_agent(model_policy, model_Vn, env_name, lam, discount, beta, train_Vn,
                                         optimizer_policy, optimizer_Vn, n_games_per_cycle, gradient_clip_policy,
                                         gradient_clip_Vn, median_stop_threshold, median_stop_patience, device)
        
        reward_sum[cycle,:] = curr_sum
        reward_len[cycle,:] = curr_len
        
        # Saving model and results            
        if save_cycles is not None and cycle%save_cycles==0:
            torch.save(model_policy.state_dict(), save_name + "_policy.pt")
            torch.save(model_Vn.state_dict(), save_name + "_Vn.pt")
            np.savetxt(save_name + "_episoderewards.csv", reward_sum[:cycle+1,:], delimiter=',')
            np.savetxt(save_name + "_episodelength.csv", reward_len[:cycle+1,:], delimiter=',')
        
        if report_updates is not None and cycle>0 and cycle%report_updates==0:
            if length:
                median_reward = np.quantile(reward_len[cycle,:], 0.5)
                q1_reward = np.quantile(reward_len[cycle,:], 0.25)
                q3_reward = np.quantile(reward_len[cycle,:], 0.75)
                print("cycle: " + str(cycle) + ' length: ' + str(median_reward) + " - q1:"+str(q1_reward)+ " - q3:"+str(q3_reward))
            else:
                median_reward = np.quantile(reward_sum[cycle,:], 0.5)
                q1_reward = np.quantile(reward_sum[cycle,:], 0.25)
                q3_reward = np.quantile(reward_sum[cycle,:], 0.75)
                print("cycle: " + str(cycle) + ' reward: ' + str(median_reward) + " - q1:"+str(q1_reward)+ " - q3:"+str(q3_reward))
        if median_stop_threshold is not None and median_stop_patience is not None and cycle >= median_stop_patience:
            if length:
                if sum(np.quantile(reward_len[cycle+1-median_stop_patience:cycle+1,:], 0.5, axis=-1) >= median_stop_threshold) >= median_stop_patience:
                    break
            else:
                if sum(np.quantile(reward_sum[cycle+1-median_stop_patience:cycle+1,:], 0.5, axis=-1) >= median_stop_threshold) >= median_stop_patience:
                    break
    return reward_sum[:cycle+1,:], reward_len[:cycle+1,:]

if __name__ == "__main__":
    from current_models import current_policy, current_Vn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='PG_settings.txt')
    parser.add_argument('--exp_name', type=str, default='mountain_cont_A3C')
    args = parser.parse_args()
    
    settings = {}
    # Reading and updating settings from the provided CSV file
    if os.path.exists(args.settings):        
        settings_table = pd.read_csv(args.settings, sep=',', header=0)
        settings = {k:v for k, v in zip(settings_table.iloc[:,0], settings_table.iloc[:,1])} 
    
    for key, value in settings.items():
        if isinstance(value, str):
            if value.lower() in ['', 'none', 'na', 'nan']:
                settings[key] = None
        elif pd.isna(value):
            settings[key] = None
        
    for key, value in settings.items():
        if not key in ["env_name", "device", "inference_device"] and value is not None:
            if key in ["train_Vn", "n_cycles", "n_games_per_cycle", "report_updates", "median_stop_patience"]:
                settings[key] = int(value)
            elif key == "length":
                settings[key] = (value.lower()=="true")
            else:
                settings[key] = float(value)
    
    model_policy = current_policy
    model_Vn = current_Vn
        
    reward_sum, reward_len = train(model_policy, model_Vn, args.exp_name, **settings)
    
    # Saving the training results and model states
    np.savetxt(args.exp_name + "_episoderewards.csv", reward_sum, delimiter=',')
    np.savetxt(args.exp_name + "_episodelength.csv", reward_len, delimiter=',')   
    torch.save(model_policy.state_dict(), args.exp_name + "_policy.pt")
    torch.save(model_Vn.state_dict(), args.exp_name + "_Vn.pt")