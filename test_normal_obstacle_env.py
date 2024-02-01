import time
from NormalStepEnv import NormalStepEnv

env = NormalStepEnv(env_name="obstacle_env")
obs = env.reset()
num_suc = 0
print("----------------------\nTest 5 random actions:\n----------------------")
for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    print(action)
    obs, r, d, i = env.step(action)
    print(obs)


