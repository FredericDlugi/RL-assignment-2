# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:13:37 2020

@author: hongh
"""

import numpy as np
import gym
import time




def epsilon_greedy(a,env,eps=0.05):
    # Input parma: 'a' : the greedy action for the currently-learned policy  
    # To do : implement the epislon-greedy alogrithm here.
    # Useful function/variable : np.random.random()/randint() ,  env.nA
    # return the action index for the current state
    
    p = # To do....First sample a value 'p' uniformly from the interval [0,1).
    if p < 1-eps:   # To think : exploit or explore
        return # to do..
    else:           # To think : exploit or explore
        return # to do..

def interact_and_record(env,policy,EPSILON):
    # This function implements the sequential interaction of the agent to environement using decaying epsilon-greedy algorithm for a complete episode
    # It also records the necessary information e.g. state,action, immediate rewards in this episode.
    
    # Initilaize the environment, returning s = S_0
    s = env.reset()     
    state_action_reward = []
    
    # start interaction
    while True:
        a = epsilon_greedy(policy[s],env,eps=EPSILON)
        # Agent interacts with the environment by taking action a in state s,\  env.step()
        # receiving successor state s_, immediate reward r, and a boolean variable 'done' telling if the episode terminates.
        # You could print out each variable to check in details.
        s_,r,done,_ = env.step(a)
        # store the <s,a,immediate reward> in list for each step
        state_action_reward.append((s,a,r))
        if done:            
            break        
        s=s_ 
    
    
    G=0
    state_action_return = []
    state_action_trajectory = []
    # TO DO : Compute the return G for each state-action pair visited
    # Hint : You need to compute the return in reversed order, first from the last state-action pair, finally to (s_0,a_0)
    # Return : (1) state_action_return = [(S_(T-1), a_(T-1), G_(T-1)), (S_(T-2), a_(T-2), G_(T-2)) ,... (S_0,a_0.G_0)]
    # (2) state_action_trajectory = [(s_0, a_0), (s_1,a_1), ... (S_(T-1)), a_(T-1))] , note:  the order is different
    # Note: even if (s_n,a_n) is encountered multiple times in an episode, here we still store them in the list, checking if it is the first appearance is done in def monte_carlo()
    return state_action_return, state_action_trajectory

    
def monte_carlo(env, N_EPISODES):
    # Initialize the random policy , useful function: np.random.choice()  env.nA, env.nS
    policy = # an 1-D array of the length = env.nS
    # To do : Intialize the Q table and number of visit per state-action pair to 0 using np.zeros()
    Q = # To do
    visit = # To do
    
    
        
    # MC approaches start learning
    for i in range(N_EPISODES):
        # epsilon-greedy exploration strategy 
        epsilon = 0.4
        # Interact with env and record the necessary info for one episode.
        state_action_return, state_action_trajectory = interact_and_record(env,policy,epsilon)
      
        count_episode_length = 0 # 
        for s,a,G in state_action_return:
            count_episode_length+= 1
            # To Do: Check whether s,a is the first appearnace and perform the update of Q values
             
            # To Do : update policy for the current state, np.argmax()
            policy[s] = # To do...    
    
    ..
    
    # Return   the finally learned policy , and the number of visits per state-action pair
    return policy, visit



if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    random_seed = 13333 # Don't change
    N_EPISODES = 150000 # Don't change
    if random_seed:
        env.seed(random_seed)
        np.random.seed(random_seed)    
    GAMMA = 1.0
    start = time.time()
    
    policy,visit = monte_carlo(env,EPSILON=1.0,N_EPISODES=N_EPISODES)
    print('TIME TAKEN {} seconds'.format(time.time()-start))
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    # Convert the policy action into arrows
    policy_arrows = np.array([a2w[x] for x in policy])
    # Display the learned policy
    print(np.array(policy_arrows).reshape([-1, 4]))