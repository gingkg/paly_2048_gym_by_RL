#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: paly_2048_gym_by_RL
@file: evaluate_and_show.py
@time: 2020/6/29
@desc: 评估模型效果和渲染显示
"""

import numpy as np
import gym
import gym_game2048
import os
from parl.utils import logger

# from DQN_CNN_2048 import Model, DQN, Agent
# from DQN_FC_2048 import Model, DQN, Agent
from PG_CNN_2048 import Model, PolicyGradient, Agent
# from PG_FC_2048 import Model, PolicyGradient, Agent

ACTION_LIST = ["↑", "↓", "→", "←"]
GAMMA = 0.99                   # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.001          # 学习率，可以从 0.001 起调，尝试增减
TAU = 0.001                    # target_model 跟 model 同步参数 的 软更新参数

def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        print(i+1)
        episode_log = {"boards": [env.get_board()], "actions": ["#"], "scores": [0]}
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = obs.swapaxes(0, 2).swapaxes(1, 2)
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, info = env.step(action)
            episode_log["boards"].append(env.get_board())
            episode_log["actions"].append(ACTION_LIST[action])
            episode_log["scores"].append(info['total_score'])
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        if render and i == 0:
            env.render(episode_log)
    return np.mean(eval_reward)


if __name__ == "__main__":
    board_size = 4
    seed = None
    binary = True
    # binary = False
    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary, extractor="cnn", penalty=-10)
    # env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary, extractor="mlp", penalty=-10)
    action_dim = len(ACTION_LIST)  # 2048:4个动作(0:上, 1:下, 2:右, 3:左)
    obs_shape = [16, 4, 4]
    # obs_shape = [16]

    model = Model(action_dim)
    # algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, tau=TAU, lr=LEARNING_RATE)
    # agent = Agent(algorithm, obs_shape, action_dim, e_greed=0.5, e_greed_decrement=0)
    algorithm = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(algorithm, obs_shape, action_dim)

    # 加载模型
    # model_path = 'models/DQN_CNN_2048/dqn_cnn_2048_model_7400.ckpt'
    # model_path = "models/DQN_FC_2048/dqn_fc_2048_model_10000.ckpt"
    model_path = "models/PG_CNN_2048/pg_cnn_2048_model_50000.ckpt"
    # model_path = "models/PG_FC_2048/pg_fc_2048_model_300.ckpt"
    if os.path.exists(model_path):
        agent.restore(model_path)

    evaluate_reward = evaluate(env, agent, render=True)
    logger.info('Test reward: {}'.format(evaluate_reward))  # 打印评估的reward

