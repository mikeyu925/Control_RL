import argparse
from itertools import count

import os, sys, random

import numpy
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from parameter import args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
directory = './exp' + script_name + args.env_name + './'


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    # def __init__(self, max_size=args.capacity):
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def fanin_init(size, fanin=None):
    # 一种比较合理的初始化网络参数https://arxiv.org/abs/1502.01852
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)
    return x.type(torch.FloatTensor)


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
    # return action
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    # return Q value
    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # 定义actor网络的online 和 target网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        # 定义critic网络的online 和 target网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        # 经验回放池 和 日志
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        # actor 和 critic 更新迭代次数
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten() # flatten()返回一个折叠成一维的数组

    def update(self):

        for it in range(args.update_iteration): # default=200
            # Sample replay buffer (s_t,s_t+1,a_t,r_t,f)
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)# default=100
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad() # 梯度清零
            critic_loss.backward() # 反向传播求梯度
            self.critic_optimizer.step()  # 更新参数

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models soft-update target Net
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        # 保存的是online网络权重
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        # 加载权重
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = DDPG(state_dim, action_dim, max_action)  # 创建一个Agent
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):  # default=10
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode): # number of games :default=100000
            total_reward = 0
            step = 0
            state = env.reset()
            for t in count():
                action = agent.select_action(state)  # 传入一个state --> actor返回一个action
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high) # exploration_noise: default=0.1， clip相当于做了一个边界处理

                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval: env.render()  # 是否show UI
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done))) # 放入经验回放池

                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step + 1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()  # 更新网络
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")


def ShowEnvInfo(env):
    print(env.action_space)  # 输出动作信息   Box(-2.0, 2.0, (1,), float32)
    # print(env.action_space.n)  # 输出动作个数
    print(env.observation_space)  # 查看状态空间  Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
    print(env.observation_space.shape[0])  # 输出状态个数 3
    print(env.observation_space.high)  # 查看状态的最高值  [1. 1. 8.]
    print(env.observation_space.low)  # 查看状态的最低值  [-1. -1. -8.]

if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # script_name = os.path.basename(__file__)

    env = gym.make(args.env_name)  # 默认环境是Pendulum-v0
    ShowEnvInfo(env)

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.shape[0]  # 动作空间维度
    max_action = float(env.action_space.high[0])
    min_Val = torch.tensor(1e-7).float().to(device)  # min value

    # directory = './exp' + script_name + args.env_name + './'

    main()
