#!/usr/bin/env/ python
import gym
import numpy as np


class SARSA_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins_d1 = 20  # Number of bins to Discretize in dim 1
        self.obs_bins_d2 = 15    # Number of bins to Discretize in dim 2 -velocity, should be odd number!
        self.obs_bins = np.array([self.obs_bins_d1,self.obs_bins_d2])
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n

        # Q-values, Initialize the Q table with 1e-7 , in the last task, you can initialize it with 0 and compare the results, for task III and question III with alpha = 1/#visit
        self.Q = np.ones((self.obs_bins[0] + 1, self.obs_bins[1] + 1, self.action_shape)) * 0# (20 x 15 x 3)
        # 
        self.visit_counts = np.zeros((self.obs_bins[0] + 1, self.obs_bins[1] + 1, self.action_shape))
        self.alpha = 0.05  # Learning rate
        self.gamma = 1.0  # Discount factor
        self.epsilon = 1.0 # Initialzation of epsilon value in epsilon-greedy

    def discretize(self, obs):
        '''A function maps the continuous state to discrete bins
        '''
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, state):
        # dicreteize the observation first
        discretized_state = self.discretize(state)
        ''' To Do:
            Implement the behavior policy (episilon greedy policy) based on the discretized_state 
            return the discrete action index
        '''
        return action


    def update_Q_table(self, obs, action, reward, done, next_obs):
        '''To do: update the Q table self.Q given each state,action ,reward... 
           No parameters for return
           Directly update the self.Q here and other necessary variables here.
        '''

        

def train(agent, env, MAX_NUM_EPISODES):
    ''' Implement one step Q-learning algorithm with decaying epsilon-greedy explroation and plot the episodic reward w.r.t. each training episode
        
        return: (1) policy, a 2-dimensional array, it is 2 dimensional since the state is 2D. Each entry stores the learned policy(action).
                (2) Q table, a 3-D array
                (3) Number of visits per state-action pair, 3D array
        Useful functions: env.step() , env.reset(), 
        Recommended : print the episodic reward per episode to check you are writing the program correctly. You can also track the best episodic reward until so far
    '''
    episodic_returns = np.zeros(MAX_NUM_EPISODES)
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        # To do : update the epsilon for decaying epsilon-greedy exploration
        agent.epsilon = ...
        # To do : initialize the state
        obs = ...
        # initialization of the following variables
        episodic_return = 0

        # To complete: one complete episode loop here.
        # (1) Select an action for the current state, using  agent.get_action(obs)
        # (2) Interact with the environment, get the necessary info
        # (3) Update the Q tables using agent.update_Q_table()
        # (4) also record the episodic cumulative reward 'episodic_return'
        # (5) Update the visit_counts per state-action pair
        
        episodic_returns[episode] = episodic_return
        if episodic_return > best_reward:
            best_reward = episodic_return   
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode, 
                                     episodic_return, best_reward, agent.epsilon))
        
    # Return the trained policy
    return policy, agent.Q.copy(), agent.visit_counts.copy(), episodic_returns


def test(agent, env, policy):
    ''' TO do : test the agent with the learned policy, the structure is very similar to train() function.
        In the test phase, we choose greedy actions, we don;t update the Q-table anymore.
        Return : episodic reward (cumulative reward in an episode)
        Constrain the maximal episodic length to be 1000 to prevent the car from getting stuck in local region.
        for local users : you can add additional env.render() after env.step(a) to see the trained result.
    '''

    return episodic_reward

if __name__ == "__main__":
    ''' 
    TO DO : You need to add code for saving the statistics and plotting the result.
    For saving statistics, you could save .npy file for episodic returns. See https://numpy.org/doc/stable/reference/generated/numpy.save.html
    And for Plotting, you write a new .py file, load these .npy files from all your group members and then plot it.
    '''
    MAX_NUM_EPISODES = 2000 
    env = gym.make('MountainCar-v0').env     # Note: the episode only terminates when cars reaches the target, the max episode length is not clipped to 200 steps.
    agent = SARSA_Learner(env)
    learned_policy, Q, visit_counts, episodic_returns = train(agent, env, MAX_NUM_EPISODES)

    # after training, test the policy 10 times.
    for _ in range(10):
        reward = test(agent, env, learned_policy)
        print("Test reward: {}".format(reward))
    env.close()