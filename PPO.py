#!/usr/bin/env python
# coding=UTF-8

from collections import namedtuple
from itertools import count

import os
import time
import numpy as np
# import matplotlib.pyplot as plt

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from environment_stage_4_ppo import Env
import rospy
from std_msgs.msg import Float32MultiArray

# 設置 TensorBoard 記錄路徑
tb = SummaryWriter('/home/user/drl_ws/src/PPO-SAC-DQN-DDPG/PPO/log')

# 訓練參數設置
gamma = 0.99  # 折扣因子
render = False  # 是否渲染環境
seed = 1  # 隨機種子
log_interval = 10  # 記錄間隔

# 狀態和動作維度
num_state = 28
num_action = 5

# 初始化環境
env = Env(num_action)
torch.manual_seed(seed)  # 設置隨機種子
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module): # 定義actor網路相關設置，狀態數量:28，動作數量:5
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

class Critic(nn.Module): # 定義critic網路相關設置，輸出價值:1
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 128

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.action_loss = 0.
        self.value_loss = 0.
        self.load_models = True  # 是否加載預訓練模型
        self.load_ep = 104
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)

        if self.load_models:
            load_model1 = torch.load("/home/user/drl_ws/src/PPO-SAC-DQN-DDPG/PPO/models/state2/749state2.pt") # 加載的模型
            self.actor_net.load_state_dict(load_model1['actor_net'])
            self.critic_net.load_state_dict(load_model1['critic_net'])
            print("加載模型:", str(self.load_ep))
            print("加載模型成功!!!!!!")

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, e):
        state = {
            'actor_net': self.actor_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_net_optimizer,
            'epoch': e
        }
        torch.save(state, "/home/user/drl_ws/src/PPO-SAC-DQN-DDPG/PPO/models/state2/" + str(e) + "state2.pt") # 儲存模型的資料夾和檔名

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，訓練 {} 次'.format(i_ep, self.training_step))

                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor_net(state[index]).gather(1, action[index])
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                action_loss = -torch.min(surr1, surr2).mean()
                self.action_loss = torch.max(action_loss)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.value_loss = torch.max(value_loss)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]

def main():
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4') # 初始化ROS Node
    # ROS發布相關資訊(透過topic)
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time = time.time()

    for e in range(750, 850): # 設置開始訓練epoch和結束epoch，例如這裡代表從750開始訓練到850
        state = env.reset()
        episode_reward_sum = 0
        done = False
        episode_step = 6000 # 每個epoch有6000步

        for t in range(episode_step):
            action, action_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state
            episode_reward_sum += reward
            pub_get_action.publish(get_action)
            if e % 1 == 0:
                agent.save_param(e)
            if t >= 600:
                rospy.loginfo("time out!")
                done = True

            if done:
                result.data = [episode_reward_sum, agent.action_loss, agent.value_loss]
                pub_result.publish(result)
                # tensorboard相關
                tb.add_scalar('Loss', episode_reward_sum, e)
                tb.add_scalar('value_loss', agent.value_loss, e)
                tb.add_scalar('action_loss', agent.action_loss, e)

                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                agent.update(e)
                # 在terminal印出相關資訊
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %.2f time: %d:%02d:%02d',
                              e, episode_reward_sum, agent.counter, t, h, m, s)
                break

if __name__ == '__main__':
    main()
    print("end")
