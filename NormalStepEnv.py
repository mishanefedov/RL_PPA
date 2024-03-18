import numpy as np
import gym
import metaworld
from gym.spaces import Box
from ObstacleEnviroments.fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2


def pretty_obs(obs):
    return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
            'goal': obs[36:39], }  # 'last_measurements': obs[18:36]}


class NormalStepEnv(gym.Env):
    def render(self, mode="human"):
        self.env.render()

    def __init__(self, env_name="reach-v2", reward_type="", multi_task=0,):
        self.env_name = env_name
        if self.env_name == "obstacle_env":
            # set environment dimensions
            self.env_dimension = [(1.05, 1.5), (0.4, 1.1), (0.4, 0.44)]
            # set environment
            self.env = FetchPickDynObstaclesEnv2()
            # set observation space
            obs = self.env.reset()['observation']
            self.observation_space = Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")
        else:
            # set environment:
            mt1 = metaworld.MT1(env_name)  # Construct the benchmark, sampling tasks
            env = mt1.train_classes[env_name]()  # Create an environment with task `pick_place`
            self.tasks = mt1.train_tasks
            self.cur_task_index = 0
            env.set_task(self.tasks[self.cur_task_index])  # Set task
            self.env = env

            # define oberservation space (copied from sawyer_xyz_env
            hand_space = Box(
                np.array([-0.525, .348, -.0525]),
                np.array([+0.525, 1.025, .7]), dtype=np.float32
            )
            gripper_low = -1.
            gripper_high = +1.
            obs_obj_max_len = 14
            obj_low = np.full(obs_obj_max_len, -np.inf)
            obj_high = np.full(obs_obj_max_len, +np.inf)
            goal_low = np.zeros(3)
            goal_high = np.zeros(3)
            self.observation_space = Box(
                np.hstack((hand_space.low, gripper_low, obj_low, goal_low)),
                np.hstack((hand_space.high, gripper_high, obj_high, goal_high)), dtype=np.float32
            )
        # other
        # define action space:
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]), dtype=np.float32
        )
        self._max_episode_length = 500
        self.number_steps = 0
        self.reward_type = reward_type

    def reset(self):
        self.number_steps = 0
        if self.env_name == "obstacle_env":
            return self.new_obs(self.env.reset())
        self.cur_task_index += 1
        if self.cur_task_index >= len(self.tasks):
            self.cur_task_index = 0
        self.env.set_task(self.tasks[self.cur_task_index])
        obs = self.env.reset()
        return self.new_obs(obs)

    def step(self, action):
        self.number_steps += 1
        obs, reward, done, info = self.env.step(action)
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        if self.env_name == "obstacle_env":
            return self.new_obs(obs), reward, done, info
        reward = -1
        if info["success"]:
            done = True
            reward = 1000
        obs = self.new_obs(obs)
        return obs, reward, done, info

    def new_obs(self, obs):
        if self.env_name == "obstacle_env":
            return obs["observation"]
        po = pretty_obs(obs)
        x = po['gripper_pos']
        x = np.append(x, po['first_obj'])
        x = np.append(x, po['second_obj'])
        x = np.append(x, po['goal'])
        return x

