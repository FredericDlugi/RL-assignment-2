#!/usr/bin/env/ python
from typing import Tuple
import gym
import numpy as np
import time
import random
import tqdm
import os


class SARSA_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins_d1 = 20  # Number of bins to Discretize in dim 1
        # Number of bins to Discretize in dim 2 -velocity, should be odd
        # number!
        self.obs_bins_d2 = 15
        self.obs_bins = np.array([self.obs_bins_d1, self.obs_bins_d2])
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n

        # Q-values, Initialize the Q table with 1e-7 , in the last task, you
        # can initialize it with 0 and compare the results, for task III and
        # question III with alpha = 1/#visit
        self.Q = np.ones(
            (self.obs_bins[0] + 1,
             self.obs_bins[1] + 1,
             self.action_shape)) * 1e-7  # (20 x 15 x 3)
        #
        self.visit_counts = np.zeros(
            (self.obs_bins[0] + 1, self.obs_bins[1] + 1, self.action_shape))
        self.alpha = 0.05  # Learning rate
        self.gamma = 1.0  # Discount factor
        self.epsilon = 1.0  # Initialzation of epsilon value in epsilon-greedy

    def discretize(self, obs) -> Tuple[int, int]:
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
        p = np.random.random()
        if p < (1 - self.epsilon):
            action = np.argmax(self.Q[discretized_state])
        else:
            action = np.random.choice(self.action_shape)
        return action

    def update_Q_table(self, obs, action, reward, done, next_obs, next_action):
        '''To do: update the Q table self.Q given each state, action ,reward...
           No parameters for return
           Directly update the self.Q here and other necessary variables here.
        '''
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)
        q_sa = self.Q[(*state, action)]
        self.Q[(*state, action)] = self.Q[(*state, action)] + (not done) * self.alpha * (reward +
                                                                            self.gamma * self.Q[(*next_state, next_action)] - self.Q[(*state, action)])


def train(agent, env, MAX_NUM_EPISODES):
    ''' Implement one step Q-learning algorithm with decaying epsilon-greedy explroation and plot the episodic reward w.r.t. each training episode

        return: (1) policy, a 2-dimensional array, it is 2 dimensional since the state is 2D. Each entry stores the learned policy(action).
                (2) Q table, a 3-D array
                (3) Number of visits per state-action pair, 3D array
        Useful functions: env.step() , env.reset(),
        Recommended : print the episodic reward per episode to check you are writing the program correctly. You can also track the best episodic reward until so far
    '''
    episodic_returns = np.zeros(MAX_NUM_EPISODES)
    best_return = -float('inf')

    pbar = tqdm.tqdm(range(MAX_NUM_EPISODES))
    for episode in pbar:
        env.reset()
        # To do : update the epsilon for decaying epsilon-greedy exploration
        agent.epsilon = 1 - episode / MAX_NUM_EPISODES
        # initialization of the following variables
        episodic_return = 0
        done = False

        # To do : initialize the state
        obs = env.state
        # (1) Select an action for the current state, using  agent.get_action(obs)
        action = agent.get_action(obs)

        while not done:
            # To complete: one complete episode loop here.
            # (2) Interact with the environment, get the necessary info
            next_obs, reward, done, _ = env.step(action)
            # (3) Update the Q tables using agent.update_Q_table()
            next_action = agent.get_action(next_obs)
            agent.update_Q_table(
                obs, action, reward, done, next_obs, next_action)
            # (4) also record the episodic cumulative reward 'episodic_return'
            episodic_return += reward
            # (5) Update the visit_counts per state-action pair
            agent.visit_counts[(*agent.discretize(obs), action)] += 1

            obs = next_obs
            action = next_action

        episodic_returns[episode] = episodic_return
        best_return = max(episodic_return, best_return)
        pbar.set_description(
            "return: {:7} best_return: {:7} eps: {:.3f}".format(
                episodic_return, best_return, agent.epsilon))

    policy = np.zeros((agent.obs_bins[0] + 1,
                       agent.obs_bins[1] + 1), dtype=int)

    for s0 in range(agent.Q.shape[0]):
        for s1 in range(agent.Q.shape[1]):
            policy[s0, s1] = np.argmax(agent.Q[s0, s1, :])

    # Return the trained policy
    return policy, agent.Q.copy(), agent.visit_counts.copy(), episodic_returns


def test(agent, env, policy):
    ''' TO do : test the agent with the learned policy, the structure is very similar to train() function.
        In the test phase, we choose greedy actions, we don;t update the Q-table anymore.
        Return : episodic reward (cumulative reward in an episode)
        Constrain the maximal episodic length to be 1000 to prevent the car from getting stuck in local region.
        for local users : you can add additional env.render() after env.step(a) to see the trained result.
    '''
    env.reset()

    obs = env.state
    episodic_return = 0
    for _ in range(1000):
        obs, reward, done, _ = env.step(policy[agent.discretize(obs)])
        env.render()
        episodic_return += reward
        if done:
            break
    return episodic_return


if __name__ == "__main__":
    '''
    TO DO : You need to add code for saving the statistics and plotting the result.
    For saving statistics, you could save .npy file for episodic returns. See https://numpy.org/doc/stable/reference/generated/numpy.save.html
    And for Plotting, you write a new .py file, load these .npy files from all your group members and then plot it.
    '''
    MAX_NUM_EPISODES = 2000
    # Note: the episode only terminates when cars reaches the target, the max
    # episode length is not clipped to 200 steps.
    env = gym.make('MountainCar-v0').env
    env.reset()
    agent = SARSA_Learner(env)
    learned_policy, Q, visit_counts, episodic_returns = train(
        agent, env, MAX_NUM_EPISODES)

    print(learned_policy)
    os.makedirs("sarsa", exist_ok=True)
    np.save(os.path.join("sarsa", "learned_policy.npy"), learned_policy)
    np.save(os.path.join("sarsa", "visits.npy"), visit_counts)
    np.save(os.path.join("sarsa", "episodic_returns.npy"), episodic_returns)

    learned_policy = np.load(os.path.join("sarsa", "learned_policy.npy"))
    # after training, test the policy 10 times.
    for _ in range(10):
        reward = test(agent, env, learned_policy)
        print("Test reward: {}".format(reward))
    env.close()
