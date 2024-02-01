# from gym.wrappers import Monitor
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO

import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from RL_PPA_monitor import RLPPAMonitor

import time


def train():
    models_dir = f"models/PPO"
    logdir = "logs"
    # timestamps = 8192
    timestamps = 2048
    number_envs_per_task = [5, 15, 30, 3, 2, 4, 3, 20, 3, 2]
    number_envs = sum(number_envs_per_task)
    # batch_size = 16384
    batch_size = 4096
    rew_type = "rew1"

    # create env
    mt10 = metaworld.MT10()
    env_array = []

    for i, (name, _) in enumerate(mt10.train_classes.items()):
        for _ in range(number_envs_per_task[i]):
            env_array.append(make_env(name, rew_type, 10, i))

    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec, "logs/PPO_0", multi_env=True, num_tasks=10)

    # create or load model
    model = PPO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=timestamps,
                batch_size=batch_size, )
    # model = PPO.load("models/PPO/534528.zip", env=env_vec, tensorboard_log=logdir)
    # safe models
    i = 0
    while True:
        print("Learning loop ", i)
        i += 1
        model = model.learn(total_timesteps=timestamps, reset_num_timesteps=False,
                            tb_log_name="PPO", )
        model.save(f"{models_dir}/{timestamps * i * number_envs}")


def make_env(name,rew_type,number_of_one_hot_tasks, one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index, teleporting=True)
    return _init


if __name__ == '__main__':
    train()
