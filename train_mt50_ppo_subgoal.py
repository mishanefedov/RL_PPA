
from stable_baselines3 import PPO

import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from RL_PPA_monitor import RLPPAMonitor


def train():
    # variables:
    models_dir = f"models/PPO"
    logdir = "logs"
    timestamps = 8192
    number_envs = 0
    indeces_with_two_envs = [9,5,4,47,46,45,43,42,41,40,3,39,35,34,33,32,30,2,29,28,27,1,17,13,12,10,0]
    number_envs_per_task = [1]*50
    for i in range(50):
        if i in indeces_with_two_envs:
            number_envs_per_task[i] = 2
            number_envs +=2
        number_envs += 1
    batch_size = 65536
    rew_type = "rew1"
    mt50 = metaworld.MT50()
    env_array = []
    for i, (name, _) in enumerate(mt50.train_classes.items()):
        print(name)
        for _ in range(number_envs_per_task[i]):
            env_array.append(make_env(name, rew_type, 50, i))

    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec, "logs/PPO_0", multi_env=True, num_tasks=50)

    # create or load model
    model = PPO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=timestamps,
                batch_size=batch_size, )
    # model = ALGO.load("models/PPO3/15360000.zip", env=env_vec,tensorboard_log=logdir)

    # safe models
    i = 0
    while True:
        print(i)
        i += 1
        model = model.learn(total_timesteps=timestamps, reset_num_timesteps=False,
                            tb_log_name="PPO", )
        model.save(f"{models_dir}/{timestamps * i * number_envs}")


def make_env(name, rew_type, number_of_one_hot_tasks, one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index)
    return _init


if __name__ == '__main__':
    train()
