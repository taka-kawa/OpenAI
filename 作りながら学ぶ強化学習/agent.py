import gym
import numpy as np
import math
import random

from brain import Brain

# CartPoleで動くエージェントクラスです、棒付き台車そのものになります
class Agent:
    def __init__(self, num_states, num_actions):
        # 課題の状態と行動の数を設定
        self.num_states = num_states     # CartPoleは状態数4を取得
        self.num_actions = num_actions        # CartPoleの行動（右に左に押す）の2を取得
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための脳を生成

    def update_q_function(self, observation, action, reward, observation_next):
        # Q関数の更新
        self.brain.update_Qtable(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        # 行動の決定
        action = self.brain.decide_action(observation, step)
        return action

