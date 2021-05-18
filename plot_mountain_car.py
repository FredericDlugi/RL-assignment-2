
import os
import numpy as np
import matplotlib.pyplot as plt
import gym

if __name__ == "__main__":
    env = gym.make('MountainCar-v0').env
    env.reset()

    learned_policy = np.load(os.path.join("sarsa", "learned_policy.npy"))
    visit_counts = np.load(os.path.join("sarsa", "visits.npy"))
    episodic_returns = np.load(os.path.join("sarsa", "episodic_returns.npy"))

    plt.suptitle("SARSA MountainCar-v0 results")
    plt.subplot(1,2,1)
    visit_counts_states = visit_counts[:,:,0] + visit_counts[:,:,1] + visit_counts[:,:,2]
    plt.title("Visits of each state")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Position (m)")
    vel_ticks = [f"{i:.2f}" for i in np.linspace(env.observation_space.low[1], env.observation_space.high[1], 6)]
    pos_ticks = [f"{i:.2f}" for i in np.linspace(env.observation_space.low[0], env.observation_space.high[0], 7)]
    plt.xticks(np.linspace(0, visit_counts.shape[1]-1, 6), vel_ticks)
    plt.yticks(np.linspace(0, visit_counts.shape[0]-1, 7), pos_ticks)
    plt.imshow(visit_counts_states.astype(float))

    plt.subplot(1,2,2)
    plt.title("Return over episodes")
    plt.xlabel("episodes")
    plt.ylabel("Cummulative Return")
    plt.plot(episodic_returns)

    plt.show()

