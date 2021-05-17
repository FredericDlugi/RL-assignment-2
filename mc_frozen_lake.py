# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:13:37 2020

@author: hongh
"""

import numpy as np
import gym
import time
import random
from typing import List, Tuple
import tqdm


def epsilon_greedy(a: int, env, eps=0.05) -> int:
    # Input param: 'a' : the greedy action for the currently-learned policy
    # return the action index for the selected action

    p = random.random()

    if p < (1 - eps):
        # we select the greedy action a according to
        # SullonBartol: Fig 5.6
        return a
    else:
        # we select a random action, this can also be a
        return np.random.choice(env.nA)


def interact_and_record(env, policy: np.ndarray,
                        eps: float, gamma: float = 0.99) -> Tuple[List, List]:
    # This function implements the sequential interaction of the agent to environement
    # using decaying epsilon-greedy algorithm for a complete episode
    # It also records the necessary information e.g. state,action, immediate
    # rewards in this episode.

    # Initilaize the environment, returning s = S_0
    s = env.reset()
    state_action_reward = []

    # start interaction
    while True:
        a = epsilon_greedy(policy[s], env, eps=eps)
        # Agent interacts with the environment by taking action a in state s,\  env.step()
        # receiving successor state s_, immediate reward r, and a boolean variable 'done'
        # telling if the episode terminates.
        # You could print out each variable to check in details.
        s_, r, done, _ = env.step(a)
        # store the <s,a,immediate reward> in list for each step

        state_action_reward.append((s, a, r))
        if done:
            break

        s = s_

    state_action_reward.reverse()

    state_action_return = []
    G = 0
    for s, a, r in state_action_reward:
        G = G * gamma + r
        state_action_return.append((s, a, G))

    return state_action_return


def monte_carlo(env, n_episodes: int,
                eps: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    # should be implemented from Fig 5.10
    # Initialize the random policy , useful function: np.random.choice()
    # env.nA, env.nS
    # an 1-D array of the length = env.nS
    policy = np.random.choice(env.nA, env.nS)
    # To do : Intialize the Q table and number of visit per state-action pair
    # to 0 using np.zeros()
    Q = np.random.random((env.nS, env.nA))

    returns = []
    for s in range(env.nS):
        returns.append([])
        for a in range(env.nA):
            returns[s].append([])

    # MC approaches start learning
    for i in tqdm.tqdm(range(n_episodes)):
        # Interact with env and record the necessary info for one episode.
        state_action_return = interact_and_record(
            env, policy, eps, gamma)

        state_action_return.reverse()
        visited = np.full((env.nS, env.nA), False)
        for s, a, g in state_action_return:
            # Check whether s,a is the first appearnace and perform the
            # update of Q values
            if not visited[s, a]:
                visited[s, a] = True
                returns[s][a].append(g)
                Q[s, a] = np.average(returns[s][a])

        for s in range(env.nS):
            policy[s] = np.argmax(Q[s, :])

    visits = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            visits[s, a] = len(returns[s][a])
    # Return the finally learned policy , and the number of visits per
    # state-action pair
    print(visits)
    print(Q)
    return policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.render()
    random_seed = 13333  # Don't change
    N_EPISODES = 15000  # Don't change
    if random_seed:
        env.seed(random_seed)
        np.random.seed(random_seed)
    epsilon = 0.4
    gamma = 0.9
    start = time.time()

    policy = monte_carlo(env, N_EPISODES, epsilon, gamma)
    print('TIME TAKEN {} seconds'.format(time.time() - start))
    a2w = {0: '<', 1: 'v', 2: '>', 3: '^'}
    # Convert the policy action into arrows
    policy_arrows = np.array([a2w[x] for x in policy])
    # Display the learned policy
    print(np.array(policy_arrows).reshape([-1, 4]))
