import time

from SubGoalEnv import SubGoalEnv,pretty_obs
import metaworld

rew_type = "rew1"
mt10 = metaworld.MT10()
env_array = []
one_hot_index = 0
for name, env_cls in mt10.train_classes.items():
    env = SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=10, one_hot_task_index=one_hot_index)
    env_array.append(env)
    one_hot_index +=1


for i,env in enumerate(env_array):
    print("Task: ",i)
    obs = env.reset()
    print(pretty_obs(obs))