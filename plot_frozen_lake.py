
import os
import numpy as np
import matplotlib.pyplot as plt
import gym
from numpy.core.fromnumeric import size

def plot_policy(learned_policy):
    holes = np.array([(1,2), (3,1), (3,2), (0,0)])
    goal = (3,0)
    learned_policy = np.rot90(learned_policy, 3)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    indicies_0 = np.argwhere(learned_policy == 0)
    plt.scatter(indicies_0[:,0], indicies_0[:,1], marker="<", label="left", s=100)
    indicies_1 = np.argwhere(learned_policy == 1)
    plt.scatter(indicies_1[:,0], indicies_1[:,1], marker="v", label="down", s=100)
    indicies_2 = np.argwhere(learned_policy == 2)
    plt.scatter(indicies_2[:,0], indicies_2[:,1], marker=">", label="right", s=100)
    indicies_3 = np.argwhere(learned_policy == 3)
    plt.scatter(indicies_3[:,0], indicies_3[:,1], marker="^", label="up", s=100)

    plt.scatter(goal[0], goal[1],marker="s", s= 500, c="red")
    plt.scatter(holes[:,0],holes[:,1],marker="s", s= 500, c="black")

    plt.xticks([])
    plt.yticks([])
    plt.xlim(left=-0.5, right=3.5)
    plt.ylim(bottom=-0.5, top=3.5)

def plot_visits(file):
    visits = np.load(file)
    plt.imshow(visits, vmin=0, vmax=320000)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

if __name__ == "__main__":

    plt.figure(figsize=(10, 3))
    plt.subplot(1,3,1)
    plt.title("Policy Iteration")
    pi_policy = np.load(os.path.join("frozen_lake_results", "pi_policy.npy"))
    plot_policy(pi_policy)
    plt.subplot(1,3,2)
    plt.title("Value Iteration")
    vi_policy = np.load(os.path.join("frozen_lake_results", "vi_policy.npy"))
    plot_policy(vi_policy)
    plt.subplot(1,3,3)
    plt.title("Monte Carlo $\epsilon=0.4$")
    mc_policy = np.load(os.path.join("frozen_lake_results", "mc_policy_40.npy"))
    plot_policy(mc_policy)
    plt.legend(loc="lower right", bbox_to_anchor=(1.45, -0.03))
    plt.savefig(os.path.join("plots", "frozenlake_policies.pdf"))

    print(pi_policy)
    print(vi_policy)
    print(mc_policy)
    plt.clf()
    plt.figure(figsize=(3, 3))
    plt.axis("equal")
    plt.title("Monte Carlo $\epsilon=0.05$")
    mc_policy_5 = np.load(os.path.join("frozen_lake_results", "mc_policy_05.npy"))
    plot_policy(mc_policy_5)

    plt.savefig(os.path.join("plots", "frozenlake_mc05.pdf"))

    plt.clf()
    plt.figure(figsize=(7, 3))
    plt.suptitle("MC visits while training")
    plt.subplot(1,2,1)
    plt.title("$\epsilon=0.4$")
    plot_visits(os.path.join("frozen_lake_results", "mc_visits_40.npy"))
    plt.subplot(1,2,2)
    plt.title("$\epsilon=0.05$")
    plot_visits(os.path.join("frozen_lake_results", "mc_visits_05.npy"))

    plt.savefig(os.path.join("plots", "frozenlake_visits_mc40_05.pdf"))