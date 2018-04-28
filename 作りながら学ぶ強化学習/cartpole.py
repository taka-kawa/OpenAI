import gym
import numpy as np
import math
import random

from agent import Agent


# 定数の設定
ENV = 'CartPole-v0'  # 使用する課題名

MAX_STEPS = 200  # 1試行のstep数
NUM_EPISODES = 1000  # 最大試行回数

# CartPoleを実行する環境のクラスです
class Environment:
    def __init__(self):
        self.env = gym.make(ENV)  # 実行する課題を設定
        self.num_states = self.env.observation_space.shape[0]  # 課題の状態と行動の数を設定
        self.num_actions = self.env.action_space.n  # CartPoleの行動（右に左に押す）の2を取得
        self.agent = Agent(self.num_states, self.num_actions)  # 環境内で行動するAgentを生成

    def run(self):
        # 実行

        complete_episodes = 0  # 195step以上連続で立ち続けた試行数
        episode_final = False  # 最後の試行フラグ

        for episode in range(NUM_EPISODES):
        # 試行数分繰り返す
            observation = self.env.reset()  # 環境の初期化
            episode_reward = 0  # エピソードでの報酬
            # 1エピソードのループ
            for step in range(MAX_STEPS):
                if episode % 30 == 0:
                    self.env.render()

                action = self.agent.get_action(observation, episode)  # 行動を求める

                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, reward_notuse, done, info_notuse = self.env.step(action)

                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    if step < 195:
                        reward = -1  # 途中でこけたら罰則として報酬-1を与える
                        self.complete_episodes = 0
                    else:
                        reward = 1  # 立ったまま終了時は報酬1を与える
                        self.complete_episodes += 1  # 連続記録を更新
                else:
                    reward = 0

                episode_reward += reward  # 報酬を追加

                # step+1の状態observation_nextを用いて,Q関数を更新する
                self.agent.update_q_function(observation, action, reward, observation_next)

                # 観測の更新
                observation = observation_next

                # 終了時の処理
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(
                        episode, step+1))
                    break

            if episode_final is True:
                # 動画を保存と描画
                break

            if self.complete_episodes >= 10:
                print('10回連続成功')
                frames = []
                episode_final = True  # 次の試行を描画を行う最終試行とする

if __name__ == "__main__":
    env = Environment()
    env.run()