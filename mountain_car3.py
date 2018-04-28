import gym
import numpy as np
import math
import random

N = 3 #モデルパラメータ数(mu:2,sigma:1)
class MountainCar3():
    def __init__(self):
        self.env = gym.make("MountainCarContinuous-v0")

    def normalize(self, state):
        high = self.env.observation_space.high  # 状態の最大値の配列
        low = self.env.observation_space.low  # 状態の最小値の配列
        return (state - low) / (high - low)  # 全ての値を0-1に正規化

    def getAction(self, sigma, mu, state, debug=False):
        """行動決定"""
        max_a = self.env.action_space.high[0]
        min_a = self.env.action_space.low[0]
        action = np.random.randn() * sigma + np.dot(mu.T, state)

        action = min(action, max_a)
        action = max(action, min_a)
        return [action]

    def NaturalActorCriric(self, L, M, T):
        """自然勾配法アルゴリズム"""

        N = 3  # モデルパラメータ数(mu:2次元。状態数に基づく sigma:1次元)

        gamma = 0.5
        alpha = 0.05

        # 政策モデルパラメータをランダムに初期化
        mu = np.random.rand(N - 1) - 0.5  # 平均。ガウス分布の軸。
        sigma = np.random.rand() * 2  # 分散。ガウス分布の片幅。

        # デザイン行列Z,報酬ベクトルqおよび
        # アドバンテージ関数のモデルパラメータwの初期化
        Z = np.zeros((M, N))
        q = np.zeros((M, 1))
        w = np.zeros((N))

        # 政策反復
        for l in range(L):
            print("episode"+str(l))
            print(mu, sigma, w)
            dr = 0
            rewards = np.empty((0, T), float)  # 報酬 M*T
            goal_count = 0
            for m in range(M):  # エピソード。標本抽出。
                # 行列derのm行目を動的確保
                der = np.zeros((N))
                # 配列rewardsのm行目を確保
                rewards = np.append(rewards, np.zeros((1, T)), axis=0)

                actions = np.zeros((T))
                states = np.zeros((N - 1, T))

                # 状態の初期化
                state = self.normalize(self.env.reset())  # 状態を初期化、0-1に正規化
                for t in range(T):  # ステップ
                    debug = False
                    if m == M - 1 and l % 5 == 0:
                        # self.env.render()
                        debug = True

                    # 行動決定
                    action = self.getAction(sigma, mu, state, debug)

                    # 行動実行、観測
                    observation, reward, done, _ = self.env.step(action)
                    state = self.normalize(observation)

                    if state[0] >= 0.5:
                        reward = 100

                    states[:,t] = state
                    actions[t] = action[0]
                    # 割引報酬和の観測
                    rewards[m, t] = reward  # デバッグ用
                    dr += (gamma ** t) * rewards[m, t]  # 政策毎

                    if done:
                        if t < T-2: # goal
                            # print("Episode %d finished after {} timesteps".format(t) % m)
                            goal_count += 1
                        break

                for t in range(T):
                    # 平均muに関する勾配の計算
                    der[0:N-1] += (actions[t] - np.dot(mu.T, states[:, t])) * states[:, t] / (sigma ** 2)
                    # 標準偏差sigmaに関する勾配の計算
                    der[-1] += ((actions[t] - np.dot(mu.T, states[:, t])) ** 2 - (sigma ** 2)) / (sigma ** 3)
                    # デザイン行列Z及び報酬ベクトルq
                    Z[m, :] += (gamma ** t) * der
                    q[m] += (gamma ** t) * (rewards[m, t])
            print(str(goal_count) + "/" + str(M))

            # r - V(s1)
            q -= dr / M
            # 最小二乗法を用いてアドバンテージ関数のモデルパラメータを推定
            Z[:, -1] = np.ones((M))

            pinv = np.linalg.pinv(np.dot(Z.T, Z))
            w = np.dot(np.dot(pinv, Z.T), q).reshape(N, )

            # wを用いてモデルパラメータを更新
            mu += alpha * w[0:N-1]
            sigma += alpha * w[N-1]

if __name__ == '__main__':
    mc = MountainCar3()

    L = 1000
    M = 200
    T = 1000

    mc.NaturalActorCriric(L, M, T)