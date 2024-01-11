
from ObstacleEnviroments.fetch.pick_dyn_obstacles2 import pretty_obs
from SubGoalEnv import SubGoalEnv

env = SubGoalEnv(env="obstacle_env", render_subactions=False, rew_type="rew1")
num_suc = 0
iterations = 100
for _ in range(iterations):
    print("---------------------------------------")
    obs = env.reset()
    print("obs: ",pretty_obs(obs))
    goal_position = env.pretty_obs(obs)["goal"]
    action = env.scale_env_pos_to_action(goal_position)
    action.append(1)
    print("action:",action)
    obs, r, d, i = env.step(action)
    if i["success"]:
        print("success")
        num_suc += 1
    else:
        print("fail")
print("---------------------------------------")
print("Number of successes ", num_suc, " of ", iterations)







