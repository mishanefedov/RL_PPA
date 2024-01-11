import time
from typing import Tuple, Dict
import numpy as np
import gym
import metaworld
from gym.spaces import Box
from GripperControl import reach, teleport, Obstacles
from ObstacleEnviroments.fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2, pretty_obs


class SubGoalEnv(gym.Env):
    """
    Created the custom subgoal environment
    """

    def __init__(self,
                 env="reach-v2",
                 render_subactions=False,
                 rew_type="",
                 number_of_one_hot_tasks=1,
                 one_hot_task_index=-1,
                 teleporting=True):
        rew_types = ["", "meta_world_rew", "rew1", "normal"]
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
        self.teleporting = teleporting
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
                one_hot = Box(np.ones(number_of_one_hot_tasks), np.ones(number_of_one_hot_tasks) * -1, dtype=np.float32)
                self.observation_space = Box(
                    np.hstack((hand_space.low, gripper_low, obj_low,
                               hand_space.low, gripper_low, obj_low, goal_low, one_hot.low)),
                    np.hstack((hand_space.high, gripper_high, obj_high,
                               hand_space.high, gripper_high, obj_high, goal_high, one_hot.high))
                    , dtype=np.float32)

    def pretty_obs(self, obs):
        if self.env_name == "obstacle_env":
            return pretty_obs(obs)
        return {'gripper_pos': obs[0:4],
                'first_obj': obs[4:11],
                'second_obj': obs[11:18],
                'goal': obs[36:39],
                'last_measurements': obs[18:36],
                "one_hot_task": obs[39:]}

    def scale_action_to_env_pos(self, action):
        action = np.clip(action, -1, 1)
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        env_pos = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            env_pos.append(
                (((action[i] - action_dimension[i][0]) * env_range) / action_range) + self.env_dimension[i][0])
        return env_pos

    def scale_env_pos_to_action(self, env_pos):
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        action = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            action.append(
                (((env_pos[i] - self.env_dimension[i][0]) * action_range) / env_range) + action_dimension[i][0])
        action = list(np.clip(action, -1, 1))
        return action

    def _change_obs(self, obs) -> [float]:
        # print("Change Obs: obs = ", obs)
        # if obstacle env change obs
        if self.env_name == "obstacle_env":
            return obs["observation"]
        # if normal env just return obs
        if self.number_of_one_hot_tasks <= 1:
            return obs
        # if multi env add one-hot
        one_hot = np.zeros(self.number_of_one_hot_tasks)
        one_hot[self.one_hot_task_index] = 1
        # print("Change Obs Retirn: obs = ", obs)
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

    def render(self, mode="human"):
        self.env.render()

    # def reset(self):
    #     if self.env_name != "obstacle_env":
    #         if self.cur_task_index >= len(self.tasks):
    #             self.cur_task_index = 0
    #         self.env.set_task(self.tasks[self.cur_task_index])
    #         self.cur_task_index += 1
    #         self.already_grasped = False
    #     self.number_steps = 0
    #     obs = self.env.reset()
    #     return self._change_obs(obs)
    
    def reset(self, **kwargs):
        # super().reset(seed=123)
        # super().reset()
        if self.env_name != "obstacle_env":
            if self.cur_task_index >= len(self.tasks):
                self.cur_task_index = 0
            self.env.set_task(self.tasks[self.cur_task_index])
            self.cur_task_index += 1
            self.already_grasped = False
        self.number_steps = 0
        obs = self.env.reset()
        # _, _, _, info = self.env.step([0, 0, 0, 0])
        # return self._change_obs(obs), info
        return self._change_obs(obs)

    def func_render_subactions(self):
        # render_subactions if render_subactions == True
        if self.render_subactions:
            self.env.render()
            time.sleep(0.05)
            
    # def action_is_target_pos(self, subgoal_pos):
    #     # compute if a given action corresponds to the target position
    #     target_pos = self.env.getTargetPos().copy()  # get target pos from env
    #     target_to_obj = (subgoal_pos - target_pos) * np.array([2., 2., 1.])  # scale to emphasize x, y axis
    #     target_to_obj = np.linalg.norm(target_to_obj)
    #     if target_to_obj < 0.05:
    #         return 1
    #     return 0

    def step(self, action):
        print("SUBGOAL ENV STEP: actions = ", action)
        start_step = time.time()
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
        # calculate if we want to teleport to the goal pos
        # action_is_goal = self.action_is_target_pos(sub_goal_pos)
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
        # init total time spend in ppa with 0
        time_in_ppa = st = numbers_no_path_found = time_in_mujoco = total_dis_offset = 0
        # tell info the inital distance between gripper and goal
        distance_to_goal = np.linalg.norm(gripper_pos - sub_goal_pos)
        initial_pos = gripper_pos
        # we check where teleportation would end
        teleport_final_pos = [gripper_pos.tolist()]
        a_star_final_pos = [gripper_pos.tolist()]
        initial_pos = gripper_pos
        # if it did not reach completely do again
        while np.linalg.norm(
                gripper_pos - sub_goal_pos) > 0.005 and max_it > 0:  # changed this as teleporting looses accuracy
            if self.env_name == "obstacle_env":
                # when obstacle env calculate sub_actions again after every step
                gripper_pos = obs["observation"][:3]
                step_size = 0.033
                obstacles = Obstacles(pretty_obs(obs["observation"]), self.env.dt)
                if self.teleporting:
                    # measure time spend in A* Search
                    st = time.time()
                    sub_actions = teleport(current_pos=gripper_pos,
                                           goal_pos=sub_goal_pos,
                                           gripper_closed=gripper_closed,
                                           obstacles=obstacles,
                                           env_dimension=self.env_dimension,
                                           step_size=step_size,
                                           distance_pruned=True)
                    time_in_ppa += time.time() - st
                    # return if the list is empty
                    if not sub_actions:
                        numbers_no_path_found += 1
                        break
                    if sub_actions is None:
                        sub_actions = [[0, 0, 0, -1]]
                    st = time.time()
                    obs, reward, done, info = self.env.teleport(sub_actions[-1], list(
                        map(lambda x, y: x + y, gripper_pos, sub_actions[-1])))
                    time_in_mujoco += time.time() - st
                else:
                    if self.teleporting is None:
                        sub_actions = teleport(current_pos=gripper_pos,
                                                        goal_pos=sub_goal_pos,
                                                        gripper_closed=gripper_closed,
                                                        obstacles=obstacles,
                                                        env_dimension=self.env_dimension,
                                                        step_size=0.01,
                                                        distance_pruned=True)
                        # print(f"teleport return {sub_actions}")
                        new_pos = [list(map(lambda x, y: x + y, gripper_pos, sub_actions[-1]))]
                        # print(f"we want to append {new_pos}")
                        teleport_final_pos += new_pos
                        # print(f"teleport is now {teleport_final_pos}")
                    # measure time spend in A* Search
                    st = time.time()
                    result = reach(current_pos=gripper_pos,
                                   goal_pos=sub_goal_pos,
                                   gripper_closed=gripper_closed,
                                   obstacles=obstacles,
                                   env_dimension=self.env_dimension,
                                   step_size=step_size,
                                   return_path=True)
                    time_in_ppa += time.time() - st
                    if result is None:
                        sub_actions = [[0, 0, 0, -1]]
                        a_star_final_pos += [gripper_pos.tolist()]
                    else:
                        (sub_actions, path) = result
                        a_star_final_pos += path
                        if not sub_actions:
                            numbers_no_path_found += 1
                    st = time.time()
                    obs, reward, done, info = self.env.step(sub_actions[0])
                    self.func_render_subactions()
                    time_in_mujoco += time.time() - st
                gripper_pos = obs[:3]

            else:
                # gripper_pos = self.env.tcp_center
                # print("SUBGOAL --- Observation obs: ", obs)
                # print("SUBGOAL --- Observation obs[:3]: ", obs[:3])
                # print("SUBGOAL --- Observation obs[observation][:3]: ", obs["observation"][:3])
                gripper_pos = obs[:3]
                step_size = 0.01
                # measure the time spend in A* Search
                if self.teleporting:
                    st = time.time()
                    sub_actions = teleport(current_pos=gripper_pos,
                                           goal_pos=sub_goal_pos,
                                           gripper_closed=gripper_closed,
                                           obstacles=obstacles,
                                           env_dimension=self.env_dimension,
                                           step_size=step_size,
                                           distance_pruned=True)
                    time_in_ppa += time.time() - st
                    if not sub_actions:
                        numbers_no_path_found += 1
                    if sub_actions is None:
                        sub_actions = [[0, 0, 0, -1]]
                    for i, a in enumerate(sub_actions):
                        st = time.time()
                        print("SUBGOAL self.env type = ", type(self.env))
                        obs, reward, done, info = self.env.teleport(sub_actions[-1], list(
                            map(lambda x, y: x + y, gripper_pos, sub_actions[-1])))
                        self.func_render_subactions()
                        time_in_mujoco += time.time() - st
                else:
                    if self.teleporting is None:
                        sub_actions = teleport(current_pos=gripper_pos,
                                                        goal_pos=sub_goal_pos,
                                                        gripper_closed=gripper_closed,
                                                        obstacles=obstacles,
                                                        env_dimension=self.env_dimension,
                                                        step_size=0.01,
                                                        distance_pruned=True)

                        #print(f"teleport return {sub_actions}")
                        new_pos = [list(map(lambda x, y: x + y, gripper_pos, sub_actions[-1]))]
                        #print(f"we want to append {new_pos} to go from {gripper_pos} to {sub_goal_pos}")
                        teleport_final_pos += new_pos
                        #print(f"teleport is now {teleport_final_pos}")
                    st = time.time()
                    result = reach(current_pos=gripper_pos,
                                   goal_pos=sub_goal_pos,
                                   gripper_closed=gripper_closed,
                                   obstacles=obstacles,
                                   env_dimension=self.env_dimension,
                                   step_size=step_size,
                                   return_path=True)
                    time_in_ppa += time.time() - st
                    if result is None:
                        sub_actions = [[0, 0, 0, -1]]
                        a_star_final_pos += [gripper_pos.tolist()]
                    else:
                        (sub_actions, path) = result
                        a_star_final_pos += path
                        if not sub_actions:
                            numbers_no_path_found += 1

                    for i, a in enumerate(sub_actions):
                        st = time.time()
                        obs, reward, done, info = self.env.step(a)
                        self.func_render_subactions()
                        time_in_mujoco += time.time() - st
                gripper_pos = obs[:3]  # gripper_pos = self.env.tcp_center

            max_it -= 1
            if self.teleporting is None:
                for n1, n1_ in zip(a_star_final_pos[-1], teleport_final_pos[-1]):
                    total_dis_offset += abs(n1 - n1_)
        info['total_dis_offset'] = total_dis_offset
        # close gripper if pick action
        if action_type == 1 and self.env_name != "obstacle_env":
            for i in range(15):
                st = time.time()
                obs, reward, done, info = self.env.step([0, 0, 0, 1])
                self.func_render_subactions()
                time_in_mujoco += time.time() - st
        # tell info how much time we spend in A*
        info['time_in_ppa'] = time_in_ppa
        # tell info how many times A* could not find a path
        info['number_no_A_path'] = numbers_no_path_found
        info['distance_to_goal'] = distance_to_goal
        info['initial_pos'] = initial_pos
        info['subgoal_pos'] = sub_goal_pos
        # info['action_is_goal_pos'] = action_is_goal
        info['a_search_path'] = a_star_final_pos
        info['teleport_path'] = teleport_final_pos
        # calculate reward
        reward, done = self._calculate_reward(reward, info, )
        self.number_steps += 1
        # set done if number of steps bigger then 20
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = 1 if not done else 0
            done = True
        else:
            info["TimeLimit.truncated"] = 0
        info['time_in_mujoco'] = time_in_mujoco
        info['time_subgoalstep_s'] = time.time() - start_step
        # terminated (bool) – Whether the agent reaches the terminal state (as defined under the MDP of the task)
        # which can be positive or negative. An example is reaching the goal state or moving into the lava from the
        # Sutton and Barton, Gridworld. If true, the user needs to call reset().
        #
        # truncated (bool) – Whether the truncation condition outside the scope of the MDP is satisfied. Typically,
        # this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used
        # to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().
        return self._change_obs(obs), reward, done, info