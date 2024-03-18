from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from NormalStepEnv import NormalStepEnv


def train():
    algo = "PPO"
    ALGO = PPO
    models_dir = f"models/{algo}"
    logdir = "logs"
    TIMESTEPS = 25000
    env = NormalStepEnv()
    env_vec = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env])
    env_vec = VecMonitor(env_vec, "logs/PPO_0")
    model = ALGO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=TIMESTEPS)
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=algo,)
        model.save(f"{models_dir}/{TIMESTEPS * iters*4}")


if __name__ == '__main__':
    train()


