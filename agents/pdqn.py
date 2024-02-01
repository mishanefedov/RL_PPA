# @author Metro
# @time 2021/10/29

"""
  Ref https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
      https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
"""

import math
import random
import torch.optim as optim
import torch.nn.functional as F
from agents.base_agent import Base_Agent
from agents.net import DuelingDQN, GaussianPolicy
from utilities.utilities import *


class P_DQN(Base_Agent):
    """
    A soft actor-critic agent for hybrid action spaces

    """

    NAME = 'P-DQN Agent'

    def __init__(self, config, env):
        Base_Agent.__init__(self, config, env)
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space[0].n
        self.action_params_dim = env.action_space[1].shape[0]

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.batch_size = self.hyperparameters['batch_size']
        self.gamma = self.hyperparameters['gamma']

        self.lr_critic = self.hyperparameters['lr_critic']
        self.lr_actor = self.hyperparameters['lr_actor']
        self.lr_alpha = self.hyperparameters['lr_alpha']
        self.tau_critic = self.hyperparameters['tau_critic']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.critic_hidden_layers = self.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = self.hyperparameters['actor_hidden_layers']
        
        # print("self lr_critic: ", self.lr_critic)
        # print("self lr_actor: ", self.lr_actor)
        # print("self lr_alpha: ", self.lr_alpha)

        self.counts = 0
        self.alpha = 0.2

        # ----  Initialization  ----
        self.critic = DuelingDQN(self.state_dim, self.action_params_dim * self.action_dim, self.critic_hidden_layers,
                                 ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.critic_target = DuelingDQN(self.state_dim, self.action_params_dim * self.action_dim, self.critic_hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.critic, target=self.critic_target)

        self.actor = GaussianPolicy(
            self.state_dim, self.action_params_dim * self.action_dim, self.actor_hidden_layers, env.action_space[1]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.target_entropy = -torch.Tensor([self.action_dim]).to(self.device).item()
        self.log_alpha = torch.tensor(-np.log(self.action_dim), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_critic)  # todo

    def select_action(self, state, train=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(-1. * self.counts / self.epsilon_decay)
        self.counts += 1
        if train:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_params, _, _ = self.actor.sample(state)
                
                if random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)

                else:
                    Q_a, _ = self.critic(state, action_params)
                    Q_a = Q_a.detach().cpu().numpy()
                    action = int(np.argmax(Q_a))
                action_params = action_params.detach().cpu().numpy()
        else:
            with torch.no_grad():
                _, _, action_params = self.actor.sample(state)
                Q_a = self.critic.forward(state, action_params)
                Q_a = Q_a.detach().cpu().numpy()
                action = int(np.argmax(Q_a))
                action_params = action_params.detach().cpu().numpy()

        return action, action_params

    def update(self, memory):
        state_batch, action_batch, action_params_batch, reward_batch, next_state_batch, done_batch = memory.sample(
            self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.IntTensor(action_batch).to(self.device).long().unsqueeze(1)
        action_params_batch = torch.FloatTensor(action_params_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        # ------------------------------------ update critic -----------------------------------------------
        with torch.no_grad():
            next_state_action_params, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            # print("--critic target--: next state batch = ", next_state_batch.shape, ", nect state action param = ", next_state_action_params.shape)
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action_params)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            q_next = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        # print("--critic--: state batch = ", state_batch.shape, ", action param = ", action_params_batch.shape)
        q1, q2 = self.critic(state_batch, action_params_batch)
        q_loss = F.mse_loss(q1, q_next) + F.mse_loss(q2, q_next)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        soft_update(self.critic_target, self.critic, self.tau_critic)

        # ------------------------------------ update actor -----------------------------------------------
        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        # min_q_pi = torch.min(q1_pi.gather(1, action_batch), q2_pi.gather(1, action_batch))
        min_q_pi = torch.min(q1_pi.mean(), q2_pi.mean())

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------ update alpha -----------------------------------------------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.critic.state_dict(), actor_path)
        torch.save(self.actor.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.critic.load_state_dict(torch.load(actor_path, actor_param_path))
        self.actor.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass
