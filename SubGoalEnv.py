import time
from typing import Tuple, Dict
import numpy as np
import gym
import metaworld
from gym.spaces import Box
# from GripperControl import reach, Obstacles
# from GripperControl import Obstacles
from ObstacleEnviroments.fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2, pretty_obs
try:
    import gripper_control_module
except:
    print("Failed to include C++ GripperControl module. Compile if with: \n\n c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` gripper_control_bindings.cpp -o gripper_control_module`python3-config --extension-suffix`")

def reach(current_pos, goal_pos, gripper_closed, env_dimension, obstacles=None, step_size= 0.01, teleporting = True) -> [[float]]:
    obstacles_list = [] if obstacles is None else obstacles
    if len(obstacles_list) == 0:
        mock_obstacle = gripper_control_module.Obstacle(
            [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], 0.0
        )
        obstacles_list.append(mock_obstacle)
    goal_pos = [float(x) for x in goal_pos]
    current_pos = current_pos.tolist()
    safety_margin=0.045

    trajectory_planner = gripper_control_module.FindTrajectory(
        current_pos,
        goal_pos,
        obstacles_list,
        env_dimension,
        step_size,
        safety_margin,
        gripper_closed
    )
    if teleporting:
        return trajectory_planner.teleportSearch()
    else:
        return trajectory_planner.aStarSearch()

class SubGoalEnv(gym.Env):
    def __init__(self, env="reach-v2", render_subactions=False, rew_type="",
                 number_of_one_hot_tasks=1, one_hot_task_index=-1):
        rew_types = ["","meta_world_rew","rew1","normal", "sparse"]
        if rew_type not in rew_types:
            raise Exception('rew_type needs to be one of: ', rew_types)
        self.env_rew = rew_type
        self.env_name = env
        self.action_space = Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        self._max_episode_length = 20
        self.number_steps = 0
        self.render_subactions = render_subactions
        self.already_grasped = False
        self.number_of_one_hot_tasks = number_of_one_hot_tasks
        self.one_hot_task_index = one_hot_task_index
        # different for each env
        if self.env_name == "obstacle_env":
            self.env_dimension = [(1.05, 1.5), (0.4, 1.1), (0.4, 0.44)]
            self.env = FetchPickDynObstaclesEnv2()
            obs = self.env.reset()['observation']
            self.observation_space = Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")
        else:
            self.env_dimension = [(-0.37, 0.31), (0.40, 0.91), (0.0, 0.31)]
            mt1 = metaworld.MT1(env)  # Construct the benchmark, sampling tasks
            env = mt1.train_classes[env]()  # Create an environment with task `pick_place`
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
            # different for multi task and non multi task env
            if number_of_one_hot_tasks == 1:
                self.observation_space = Box(
                    np.hstack((hand_space.low, gripper_low, obj_low,
                               hand_space.low, gripper_low, obj_low, goal_low)),
                    np.hstack((hand_space.high, gripper_high, obj_high,
                               hand_space.high, gripper_high, obj_high, goal_high))
                    , dtype=np.float32)
            else:
                one_hot = Box(np.ones(number_of_one_hot_tasks), np.ones(number_of_one_hot_tasks)*-1, dtype=np.float32)
                self.observation_space = Box(
                    np.hstack((hand_space.low, gripper_low, obj_low,
                               hand_space.low, gripper_low, obj_low, goal_low,one_hot.low)),
                    np.hstack((hand_space.high, gripper_high, obj_high,
                               hand_space.high, gripper_high, obj_high, goal_high,one_hot.high))
                    , dtype=np.float32)

    def pretty_obs(self, obs):
        if self.env_name == "obstacle_env":
            return pretty_obs(obs)
        return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
                'goal': obs[36:39], 'last_measurements': obs[18:36], "one_hot_task": obs[39:]}

    def scale_action_to_env_pos(self,action):
        action = np.clip(action, -1, 1)
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        env_pos = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            env_pos.append((((action[i] - action_dimension[i][0]) * env_range) / action_range) + self.env_dimension[i][0])
        return env_pos

    def scale_env_pos_to_action(self,env_pos):
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        action = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            action.append((((env_pos[i] - self.env_dimension[i][0]) * action_range) / env_range) + action_dimension[i][0])
        action = list(np.clip(action, -1, 1))
        return action

    def _change_obs(self, obs) ->[float]:
        # if obstacle env change obs
        if self.env_name == "obstacle_env":
            return obs["observation"]
        # if normal env just return obs
        if self.number_of_one_hot_tasks <= 1:
            return obs
        # if multi env add one-hot
        one_hot = np.zeros(self.number_of_one_hot_tasks)
        one_hot[self.one_hot_task_index] = 1
        return np.concatenate([obs, one_hot])

    def _calculate_reward(self, re, info: Dict[str, bool]) -> (int, bool):
        done = False
        if self.env_rew in ["normal",""]:
            # reward with done, but with regular environment reward
            if info['success']:
                return re, True
            return re, False
        if self.env_rew in ["meta_world_rew"]:
            # use reward from metaworld and only done if 20 steps
            return re, done
        if self.env_rew == "rew1":
            # mapped reward to [-1,0]
            if info['success']:
                return 0, True
            return (-1 + (re/10)), False
        if self.env_rew == "sparse":
            # give reward only if success
            if info['success']:
                return 1, True
            return 0, False

    def render(self, mode="human"):
        self.env.render()

    def reset(self):
        if self.env_name != "obstacle_env":
            if self.cur_task_index >= len(self.tasks):
                self.cur_task_index = 0
            self.env.set_task(self.tasks[self.cur_task_index])
            self.cur_task_index += 1
            self.already_grasped = False
        self.number_steps = 0
        obs = self.env.reset()
        return self._change_obs(obs)

    def func_render_subactions(self):
        # render_subactions if render_subactions == True
        if self.render_subactions:
            self.env.render()
            time.sleep(0.05)

    def step(self, action):
        # get kind of action: "hold"=0, "grasp"=1
        action_type = 0
        gripper_closed = True
        if action[3] > 0:
            action_type = 1
            gripper_closed = False

        # transform action into coordinates
        sub_goal_pos = self.scale_action_to_env_pos(action)

        # create initial obs,
        obs, reward, done, info = self.env.step([0, 0, 0, 0])
        # open gripper if picking,
        if action_type == 1:
            # actions need to be performed several times, because otherwise we can't guarantee that the gripper is open
            for i in range(15):
                obs, reward, done, info = self.env.step([0, 0, 0, -1])
                self.func_render_subactions()

        if self.env_name == "obstacle_env":
            gripper_pos = obs["observation"][:3]
            obstacles = Obstacles(pretty_obs(obs["observation"]), self.env.dt)
            max_it = 100
        else:
            obstacles = None
            gripper_pos = self.env.tcp_center
            max_it = 3

        # if it did not reach completely do again
        while np.linalg.norm(gripper_pos - sub_goal_pos) > 0.0005 and max_it > 0:
            if self.env_name == "obstacle_env":
                # when obstacle env calculate sub_actions again after every step
                gripper_pos = obs["observation"][:3]
                step_size = 0.033
                obstacles = Obstacles(pretty_obs(obs["observation"]), self.env.dt)
                sub_actions = reach(current_pos=gripper_pos, goal_pos=sub_goal_pos,
                                    gripper_closed=gripper_closed, obstacles=obstacles,
                                    env_dimension=self.env_dimension,step_size=step_size)
                if not sub_actions:
                    break
                if sub_actions is None:
                    sub_actions = [[0, 0, 0, -1]]
                obs, reward, done, info = self.env.step(sub_actions[0])
                self.func_render_subactions()
            else:
                # else just calculate it ones
                gripper_pos = self.env.tcp_center
                step_size = 0.01
                sub_actions = reach(current_pos=gripper_pos, goal_pos=sub_goal_pos, gripper_closed=gripper_closed,
                                    obstacles=obstacles, env_dimension=self.env_dimension, step_size=step_size)
                if sub_actions is None:
                    sub_actions = [[0, 0, 0, -1]]
                for i, a in enumerate(sub_actions):
                    obs, reward, done, info = self.env.step(a)
                    self.func_render_subactions()
            max_it -= 1
        # close gripper if pick action
        if action_type == 1 and self.env_name != "obstacle_env":
            for i in range(15):
                obs, reward, done, info = self.env.step([0, 0, 0, 1])
                self.func_render_subactions()

        # calculate reward
        reward, done = self._calculate_reward(reward, info,)
        self.number_steps += 1
        # set done if number of steps bigger then 20
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        return self._change_obs(obs), reward, done, info
