
import os
import numpy as np
import matplotlib.pyplot as plt
import gym


def save_visits(folder):
    show_visits(folder)
    plt.savefig(os.path.join("plots", f"{folder}_visits.pdf"))
    plt.clf()

def show_visits(folder):
    visit_counts = np.load(
        os.path.join(
            "mountain_car_results",
            folder,
            "visits.npy"))
    visit_counts_states = visit_counts[:, :, 0] + \
        visit_counts[:, :, 1] + visit_counts[:, :, 2]

    vel_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[1],
            env.observation_space.high[1],
            6)]
    pos_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            7)]
    plt.yticks(np.linspace(0, visit_counts.shape[1] - 1, 6), vel_ticks)
    plt.xticks(np.linspace(0, visit_counts.shape[0] - 1, 7), pos_ticks)
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Position (m)")
    plt.imshow(np.rot90(visit_counts_states.astype(float)))
    color_ticks = list(range(0, int(np.max(visit_counts_states)), 5_000)) + [np.max(visit_counts_states)]
    plt.colorbar(ticks=color_ticks)


def save_policy(folder):
    learned_policy = np.load(
        os.path.join(
            "mountain_car_results",
            folder,
            "learned_policy.npy"))

    indicies_1 = np.argwhere(learned_policy == 1)
    plt.scatter(indicies_1[:, 0], indicies_1[:, 1],
                marker=".", label="no push")
    indicies_0 = np.argwhere(learned_policy == 0)
    plt.scatter(indicies_0[:, 0], indicies_0[:, 1],
                marker="<", label="push left")
    indicies_2 = np.argwhere(learned_policy == 2)
    plt.scatter(indicies_2[:, 0], indicies_2[:, 1],
                marker=">", label="push right")

    vel_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[1],
            env.observation_space.high[1],
            6)]
    pos_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            7)]
    plt.yticks(np.linspace(0, learned_policy.shape[1] - 1, 6), vel_ticks)
    plt.xticks(np.linspace(0, learned_policy.shape[0] - 1, 7), pos_ticks)
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Position (m)")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))

    plt.savefig(os.path.join("plots", f"{folder}_learned_policy.pdf"))
    plt.clf()


if __name__ == "__main__":
    env = gym.make('MountainCar-v0').env
    env.reset()

    folder = "sarsa"
    save_policy(folder)
    sarsa_episodic_returns = np.load(
        os.path.join(
            "mountain_car_results",
            folder,
            "episodic_returns.npy"))
    save_visits(folder)

    folder = "q_learning"
    save_policy(folder)
    qlearn_episodic_returns = np.load(
        os.path.join(
            "mountain_car_results",
            folder,
            "episodic_returns.npy"))
    save_visits(folder)

    folder = "q_learning_decay_alpha"
    save_policy(folder)
    qlearn_decay_alpha_episodic_returns = np.load(os.path.join(
        "mountain_car_results", folder, "episodic_returns.npy"))
    save_visits(folder)

    folder = "q_learning_decay_alpha_q-10000000"
    qlearn_decay_alpha_q1e7 = np.load(os.path.join(
        "mountain_car_results", folder, "episodic_returns.npy"))

    max_sarsa_episodic_returns = []
    max_qlearn_episodic_returns = []
    max_qlearn_decay_alpha_episodic_returns = []
    max_qlearn_decay_alpha_q1e7 = []
    lower_index = 0
    for i in range(1, sarsa_episodic_returns.shape[0]):
        lower_index = max(0, i - 50)
        max_sarsa_episodic_returns.append(
            np.average(sarsa_episodic_returns[lower_index:i]))
        max_qlearn_episodic_returns.append(
            np.average(qlearn_episodic_returns[lower_index:i]))
        max_qlearn_decay_alpha_episodic_returns.append(
            np.average(qlearn_decay_alpha_episodic_returns[lower_index:i]))
        max_qlearn_decay_alpha_q1e7.append(
            np.average(qlearn_decay_alpha_q1e7[lower_index:i]))

    plt.clf()
    plt.title("Return over episodes (50 episode average)")
    plt.xlabel("episodes")
    plt.ylabel("Cummulative Return")
    generations = np.arange(1, sarsa_episodic_returns.shape[0])
    plt.scatter(
        generations,
        max_qlearn_decay_alpha_episodic_returns,
        label="Q-Learning $\\alpha=1/N_{visit}$",
        s=1)
    plt.scatter(
        generations,
        max_qlearn_decay_alpha_q1e7,
        label="Q-Learning $\\alpha=1/N_{visit}$ $Q_{init}=-1E7$",
        s=1)
    plt.scatter(generations, max_qlearn_episodic_returns,
                label="Q-Learning $\\alpha=0.05$", s=1)
    # plt.xlim(right=500, left=0)
    # plt.ylim(bottom=-2000)
    plt.legend()

    plt.savefig(os.path.join("plots", "return_q_learning.pdf"))

    plt.clf()
    plt.figure(figsize=(10,8))
    plt.title("Return over episodes (50 episode average)")
    plt.xlabel("episodes")
    plt.ylabel("Cummulative Return")
    generations = np.arange(1, sarsa_episodic_returns.shape[0])
    plt.scatter(generations, max_sarsa_episodic_returns, label="SARSA", s=1)
    plt.scatter(
        generations,
        max_qlearn_episodic_returns,
        label="Q-Learning",
        s=1)
    plt.legend()

    plt.savefig(os.path.join("plots", "return_over_generations.pdf"))
