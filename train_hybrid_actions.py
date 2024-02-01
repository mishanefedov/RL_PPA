# @author Metro
# @time 2021/11/24
import os.path

import gym
from agents.pdqn import P_DQN
from utilities.memory import ReplayBuffer
from utilities.utilities import *
from utilities.route_generator import generate_routefile

# Config Imports
import torch
from config_hybrid_action_space import Config

# Metaword Training Imports
from SubGoalEnv import SubGoalEnv
import metaworld
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from RL_PPA_monitor import RLPPAMonitor
import numpy as np
from typing import Tuple


config = Config()
config.seed = 123456
config.train = True
config.evaluate = False
config.evaluate_internal = 5
# config.environment = 'FreewheelingIntersection-v1'
config.environment = 'reach-v2'
config.file_to_save = 'results_hybrid_action_space/'
config.save_model = True
config.standard_deviation_results = 1.0
config.randomise_random_seed = True
config.save_freq = 20
config.simulations_num = 10
config.rolling_score_window = 5
config.runs_per_agent = 3
config.agent_name = 'P-DQN'
config.use_GPU = True
config.ceil = True
config.demand = [
    [1. / 22, 1. / 20, 1. / 21, 1. / 18, 1. / 16, 1. / 14, 1. / 13, 1. / 21, 1. / 20, 1. / 21, 1. / 19, 1. / 18],
    [1. / 20, 1. / 21, 1. / 18, 1. / 13, 1. / 16, 1. / 12, 1. / 12, 1. / 19, 1. / 13, 1. / 11, 1. / 16, 1. / 18]
]

config.env_parameters = {
    'cells': 32,
    'lane_length_high': 240.,
    'speed_high': 100.,
    'edge_ids': ['north_in', 'east_in', 'south_in', 'west_in'],
    'vehicles_types': ['NW_right', 'NS_through', 'NE_left',
                       'EN_right', 'EW_through', 'ES_left',
                       'SE_right', 'SN_through', 'SW_left',
                       'WS_right', 'WE_through', 'WN_left'],
    'yellow': 3,
    'simulation_steps': 3600,
    'n_steps': 5,
    'alpha': 0.2,  # TODO

}
config.hyperparameters = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epsilon_initial': 0.3,
    'epsilon_final': 0,
    'epsilon_decay': 3000,
    'replay_memory_size': 1e5,
    'batch_size': 128,
    'gamma': 0.98,
    'lr_critic': 1e-3,
    'lr_actor': 1e-4,
    'lr_alpha': 1e-2,
    'tau_critic': 0.01,
    'tau_actor': 0.01,
    'critic_hidden_layers': (256, 128, 64),
    'actor_hidden_layers': (256, 128, 64),
    'updates_per_step': 1,
    'maximum_episodes': 4000,
    'alpha': 0.2,
}

class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        # generate_routefile(seed=config.seed, demand=config.demand)
        # self.env = gym.make(config.environment)
        
        mt10 = metaworld.MT10()
        
        self.env = self.make_env(name='reach-v2', rew_type='rew1', number_of_one_hot_tasks=10, one_hot_task_index=1)

        # Agent
        self.agent = P_DQN(config, self.env)

        # Memory
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.batch_size = config.hyperparameters['batch_size']
        self.updates_per_step = config.hyperparameters['updates_per_step']
        self.memory = ReplayBuffer(self.replay_memory_size)

        self.total_steps = 0
        self.total_updates = 0

        self.save_freq = config.save_freq
        self.file_to_save = config.file_to_save
        self.maximum_episodes = config.hyperparameters['maximum_episodes']

        self.train = config.train
        self.evaluate = config.evaluate
        self.evaluate_internal = config.evaluate_internal

        self.agent_to_color_dictionary = config.agent_to_color_dictionary
        self.standard_deviation_results = config.standard_deviation_results

        self.colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        self.color_idx = 0

        self.rolling_score_window = config.rolling_score_window
        self.runs_per_agent = config.runs_per_agent
        self.agent_name = config.agent_name
        self.ceil = config.ceil


    def make_env(self, name, rew_type, number_of_one_hot_tasks, one_hot_task_index):
        # def _init():
        #     return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
        #                     one_hot_task_index=one_hot_task_index)
        return SubGoalEnv(
                env=name,
                rew_type=rew_type,
                number_of_one_hot_tasks=number_of_one_hot_tasks,
                one_hot_task_index=one_hot_task_index
            )
    
    def train_agent(self):
        """

        :return:
        """

        rolling_scores_for_diff_runs = []
        file_to_save_actor = os.path.join(self.file_to_save, 'actor/')
        file_to_save_actor_param = os.path.join(self.file_to_save, 'actor_param/')
        file_to_save_runs = os.path.join(self.file_to_save, 'runs_1/')
        file_to_save_rolling_scores = os.path.join(self.file_to_save, 'rolling_scores/')
        os.makedirs(file_to_save_actor, exist_ok=True)
        os.makedirs(file_to_save_actor_param, exist_ok=True)
        os.makedirs(file_to_save_runs, exist_ok=True)
        os.makedirs(file_to_save_rolling_scores, exist_ok=True)

        for run in range(self.runs_per_agent):
            game_full_episodes_scores = []
            game_full_episodes_rolling_scores = []

            for i_episode in range(self.maximum_episodes):

                episode_score = []
                episode_steps = 0
                done = 0
                state = self.env.reset()  # n_steps

                while not done:
                    if len(self.memory) > self.batch_size:
                        action, action_params = self.agent.select_action(state, self.train)
                        if self.ceil:
                            action_params = np.ceil(action_params).squeeze(0)
                            
                        if action == 0:
                            # action_params = action_params[[0,1,2]]
                            action_for_env = [action, action_params[[0,1,2]]]
                        elif action == 1:
                            # action_params = action_params[[3,4,5]]
                            action_for_env = [action, action_params[[3,4,5]]]
                        
                        action_for_env = [action, action_params]

                        for i in range(self.updates_per_step):
                            self.agent.update(self.memory)
                            self.total_updates += 1
                    else:
                        action_params = np.random.uniform(low=-1, high=1, size=6)
                        action = np.random.randint(1, size=1)[0]
                        if action == 0:
                            # action_params = action_params[[0,1,2]]
                            action_for_env = [action, action_params[[0,1,2]]]
                        elif action == 1:
                            # action_params = action_params[[3,4,5]]
                            action_for_env = [action, action_params[[3,4,5]]]
                        else:
                            print("Action unknown: action = ", action)
                            exit(0)
                        
                    # print("action for env: ", action_for_env)
                    next_state, reward, done, info = self.env.step(action_for_env)

                    episode_steps += 1
                    episode_score.append(info['success'])

                    self.total_steps += 1
                    self.memory.push(state, action, action_params, reward, next_state, done)

                    state = next_state
                    
                if self.save_freq > 0 and i_episode % self.save_freq == 0:
                    actor_path = os.path.join(file_to_save_actor, 'episode{}'.format(i_episode))
                    actor_param_path = os.path.join(file_to_save_actor_param, 'episode{}'.format(i_episode))
                    self.agent.save_models(actor_path, actor_param_path)
                    
                # print("Episode Score: ", episode_score)
                # print("Episode Steps: ", episode_steps)
                episode_score_so_far = np.mean(episode_score)
                game_full_episodes_scores.append(episode_score_so_far)
                game_full_episodes_rolling_scores.append(
                    np.mean(game_full_episodes_scores[-1 * self.rolling_score_window:]))

                print("Episode: {}, total steps:{}, episode steps:{}, scores:{}".format(
                    i_episode, self.total_steps, episode_steps, episode_score_so_far))

                # self.env.close()
                file_path_for_pic = os.path.join(file_to_save_runs, 'episode{}_run{}.jpg'.format(i_episode, run))
                visualize_results_per_run(agent_results=game_full_episodes_scores,
                                          agent_name=self.agent_name,
                                          save_freq=1,
                                          file_path_for_pic=file_path_for_pic)
                rolling_scores_for_diff_runs.append(game_full_episodes_rolling_scores)

        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'rolling_scores.jpg')
        visualize_overall_agent_results(agent_results=rolling_scores_for_diff_runs,
                                        agent_name=self.agent_name,
                                        show_mean_and_std_range=True,
                                        agent_to_color_dictionary=self.agent_to_color_dictionary,
                                        standard_deviation_results=1,
                                        file_path_for_pic=file_path_for_pic
                                        )


trainer = Train_and_Evaluate(config=config)

trainer.train_agent()