#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: paly_2048_gym_by_RL
@file: PG_FC_2048.py
@time: 2020/6/29
@desc: 
"""

# 导入依赖
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
import random
import collections
import time
import pickle
import gym_game2048

# 设置超参数
LEARNING_RATE = 1e-3
GAMMA = 0.99                    # reward 的衰减因子，一般取 0.9 到 0.999 不等

TRAIN_TOTAL_EPISODES = 50000    # 总训练步数
TEST_EVERY_EPISODES = 100       # 每个N步评估一下算法效果，每次评估5个episode求平均reward


class Model(parl.Model):
    def __init__(self, act_dim):
        # 配置model
        self.fc1 = layers.fc(size=128, act="softmax")
        self.fc2 = layers.fc(size=128, act="softmax")
        self.fc3 = layers.fc(size=128, act="softmax")
        self.fc4 = layers.fc(size=act_dim, act="softmax")

    def forward(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        # 组装Q网络
        h = self.fc1(obs)
        h = self.fc2(h)
        h = self.fc3(h)
        out = self.fc4(h)
        return out


# from parl.algorithms import PolicyGradient # 也可以直接从parl库中导入PolicyGradient算法，无需重复写算法
class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm

        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_prob = self.model(obs)  # 获取输出动作概率
        # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        log_prob = layers.reduce_sum(
            -1.0 * layers.log(act_prob) * layers.one_hot(
                action, act_prob.shape[1]),
            dim=1)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)  # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            max_socre = env.get_board().max()
            break

    return obs_list, action_list, reward_list, max_socre


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        episode_log = {"boards": [env.get_board()], "actions": ["#"], "scores": [0]}
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, info = env.step(action)
            episode_log["boards"].append(env.get_board())
            episode_log["actions"].append(ACTION_LIST[action])
            episode_log["scores"].append(info['total_score'])
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        if render:
            env.render(episode_log)
    return np.mean(eval_reward)


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


if __name__ == "__main__":
    # 创建环境和Agent，创建经验池，启动训练，保存模型
    board_size = 4
    seed = None
    binary = False
    ACTION_LIST = ["↑", "↓", "→", "←"]
    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary, extractor="mlp", penalty=-10)
    action_dim = len(ACTION_LIST)  # 2048:4个动作(0:上, 1:下, 2:右, 3:左)
    obs_shape = [16]

    # 根据parl框架构建agent
    # 嵌套Model, DQN, Agent构建 agent
    model = Model(action_dim)
    algorithm = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(algorithm, obs_shape, action_dim)

    # 加载全局episode
    episode_path = 'models/PG_FC_2048/global_episodes.pkl'
    if not os.path.exists(episode_path):
        with open(episode_path, "wb") as f:
            global_episodes = 0
            pickle.dump(global_episodes, f)

    with open(episode_path, "rb") as f:
        global_episodes = pickle.load(f)

    # 加载模型
    model_path = "models/PG_FC_2048/pg_fc_2048_model_300.ckpt"
    if os.path.exists(model_path):
        agent.restore(model_path)

    total_episodes = 0
    while total_episodes < TRAIN_TOTAL_EPISODES:
        obs_list, action_list, reward_list, max_socre = run_episode(env, agent)
        total_episodes += 1
        if total_episodes % 10 == 0:
            logger.info('Episode：{} Steps: {}  Max Socre: {} Reward: {}'.format
                        (total_episodes, len(action_list), max_socre, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list, gamma=GAMMA)

        agent.learn(batch_obs, batch_action, batch_reward)

        if total_episodes % TEST_EVERY_EPISODES == 0:  # 每隔一定step数，评估一次模型
            global_episodes = global_episodes + TEST_EVERY_EPISODES

            evaluate_reward = evaluate(env, agent, render=False)
            logger.info('Episode：{} , Test reward: {}'.format(global_episodes, evaluate_reward))  # 打印评估的reward

            # 每评估一次，就保存一次模型，以训练的step数命名
            agent.save('models/PG_FC_2048/pg_fc_2048_model_{}.ckpt'.format(global_episodes))

            with open(episode_path, "wb") as f:
                pickle.dump(global_episodes, f)










