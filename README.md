# RL-assignment-2

Authors:
- Franek Stark
- Robin Luckey
- Moritz Gerwin
- Frederic Dlugi

Reinforcement Learning Assignment 2
Report-Link [https://www.overleaf.com/2554662698pqmghkbkrmmt]

# Anaconda-Environment installation

1.  Install newest version of Anaconda

2.  Open an Anaconda-Terminal and navigate to this repo

3.  Run the following commands.
    ```
    conda config --add channels conda-forge
    conda create --name RL --file req.txt
    conda activate RL
    ```

4.  Test your installation by running
    ```
    python installation_test.py
    ```

# Manual environment installation

The Anaconda environment was created with the following commands.
```
conda create --name RL python=3.8
conda activate RL
conda install gym
conda install pybullet
conda install box2d-py
conda install atari_py
conda install matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install tqdm
conda install autopep8
```
