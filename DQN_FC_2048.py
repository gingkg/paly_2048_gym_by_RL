#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: paly_2048_gym_by_RL
@file: DQN_FC_2048.py
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
LEARN_FREQ = 10                 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000            # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 2000       # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 256                # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99                   # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.01        # 学习率，可以从 0.001 起调，尝试增减
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数

TRAIN_TOTAL_EPISODES = 5000   # 总训练步数
TEST_EVERY_EPISODES = 500    # 每个N步评估一下算法效果，每次评估5个episode求平均reward


class Model(parl.Model):
    def __init__(self, act_dim):
        # 配置model
        self.fc1 = layers.fc(size=128, act="leaky_relu")
        self.fc2 = layers.fc(size=128, act="leaky_relu")
        self.fc3 = layers.fc(size=128, act="leaky_relu")
        self.fc4 = layers.fc(size=act_dim, act="leaky_relu")
    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        # 组装Q网络
        h = self.fc1(obs)
        h = self.fc2(h)
        h = self.fc3(h)
        Q = self.fc4(h)
        return Q


class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, tau=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)  # 断点调试和类型判断，报错后面的就不执行
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]，独热编码有好处
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.5,
                 e_greed_decrement=0):
        # assert isinstance(obs_dim, int)
        # assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)

        self.global_step = 0
        self.update_target_steps = 20  # 每隔200个training steps再把model的参数复制到target_model中

        episode_path = 'models/DQN_FC_2048/global_episodes.pkl'
        if not os.path.exists(episode_path):
            with open(episode_path, "wb") as f:
                global_episodes = 0
                pickle.dump(global_episodes, f)
        with open(episode_path, "rb") as f:
            global_episodes = pickle.load(f)

        e_greed = max(0.1, e_greed - (global_episodes // 2500) * 0.1)

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=self.obs_dim, dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)  # 删除0维度
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络

        self.alg.sync_target()

        return cost


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)


def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            max_socre = env.get_board().max()
            break
    return total_reward, step, max_socre


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


if __name__ == "__main__":
    # 创建环境和Agent，创建经验池，启动训练，保存模型
    board_size = 4
    seed = None
    binary = False
    ACTION_LIST = ["↑", "↓", "→", "←"]
    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary, extractor="mlp", penalty=-10)
    action_dim = len(ACTION_LIST)  # 2048:4个动作(0:上, 1:下, 2:右, 3:左)
    obs_shape = [16]

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    # 嵌套Model, DQN, Agent构建 agent
    model = Model(action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, tau=TAU, lr=LEARNING_RATE)
    agent = Agent(algorithm, obs_shape, action_dim, e_greed=0.3, e_greed_decrement=0)



    # 加载全局episode
    episode_path = 'models/DQN_FC_2048/global_episodes.pkl'
    if not os.path.exists(episode_path):
        with open(episode_path, "wb") as f:
            global_episodes = 0
            pickle.dump(global_episodes, f)

    with open(episode_path, "rb") as f:
        global_episodes = pickle.load(f)

    # 加载模型
    model_path = "models/DQN_FC_2048/dqn_fc_2048_model_10000.ckpt"
    if os.path.exists(model_path):
        agent.restore(model_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    print(len(rpm))

    total_episodes = 0
    while total_episodes < TRAIN_TOTAL_EPISODES:
        train_reward, steps, max_socre = run_episode(env, agent, rpm)
        total_episodes += 1
        logger.info('Episode：{} Steps: {}  Max Socre: {} Reward: {}'.format(total_episodes, steps, max_socre,train_reward))  # 打印训练reward

        if total_episodes % TEST_EVERY_EPISODES == 0:  # 每隔一定step数，评估一次模型
            global_episodes = global_episodes + TEST_EVERY_EPISODES

            evaluate_reward = evaluate(env, agent, render=False)
            logger.info('Episode：{} , Test reward: {}'.format(global_episodes, evaluate_reward))  # 打印评估的reward

            # 每评估一次，就保存一次模型，以训练的step数命名
            agent.save('models/DQN_FC_2048/dqn_fc_2048_model_{}.ckpt'.format(global_episodes))

            with open(episode_path, "wb") as f:
                pickle.dump(global_episodes, f)













