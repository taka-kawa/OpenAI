import gym
import numpy as np
import math
import random

env = gym.make('Pendulum-v0')

class Pendulum():
    def __init__(self):
        self.__N = 4  # モデルパラメータ数(mu:3,sigma:1)
        self.__mu = np.random.random((self.__N - 1))
        self.__sigma = np.random.random((1))
        self.__min = -2
        self.__max = 2

    def __get_action(self, observation):
        # a = np.random.normal(np.dot(self.__mu.T, observation), self.__sigma, 1)
        a = np.random.randn() * self.__sigma + np.dot(self.__mu.T, observation)
        a = min(a, self.__max)
        a = max(a, self.__min)

        return a

    def learn(self, L=100, M=100, T=200, ALPHA=0.3, GAMMA=0.1):
        # 政策反復
        for l in range(L):
            print("policy " + str(l))
            print(self.__mu, self.__sigma)
            drs = np.zeros(M)
            der = np.zeros([M, self.__N])  # パラメータ
            for m in range(M):
                observation = env.reset()
                continuous = 0
                for t in range(T):
                    env.render()

                    action = [0]
                    # 行動決定
                    action[0] = self.__get_action(observation)

                    # 行動
                    observation, reward, done, info = env.step(action)
                    observation = np.reshape(observation, (self.__N-1,))
                    # μ計算
                    der[m][0:self.__N - 1] += (action[0] - np.dot(self.__mu.T, observation) * observation) / (
                    self.__sigma ** 2)
                    # σ計算
                    der[m][self.__N - 1] += (
                                            (action[0] - np.dot(self.__mu.T, observation)) ** 2 - self.__sigma ** 2) / (
                                            self.__sigma ** 3)
                    # 割引報酬和計算
                    drs[m] += GAMMA ** (t) * reward

                    if reward > -0.1:
                        continuous += 1
                        if continuous > 15:
                            print("Episode %d finished after {} timesteps".format(t) % m)
                    else :
                        continuous = 0
                    if done:
                        if t < T - 1:  # goal
                            print("Episode %d finished after {} timesteps".format(t) % m)
                        break

            # baseline
            baseLine = np.dot(drs,np.diag(np.dot(der,der.T)))/np.trace(np.dot(der,der.T))

            # 勾配推定
            der_J = (np.dot((drs - baseLine), der)) / M
            print(der_J)
            # モデルパラメータの推定
            self.__mu += ALPHA * der_J[0:self.__N - 1]
            self.__sigma += ALPHA * der_J[self.__N - 1]


if __name__ == '__main__':
    p = Pendulum()
    p.learn(M=100)