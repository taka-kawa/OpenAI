import gym
import numpy as np
import math
import random

env = gym.make('MountainCarContinuous-v0')

class MountainCar2():
    def __init__(self):
        self.__N = 3 #モデルパラメータ数(mu:2,sigma:1)
        self.__mu = np.random.random((self.__N-1))-0
        self.__sigma = np.random.random((1))*10

    def __get_action(self, observation):
        max_a = env.action_space.high
        min_a = env.action_space.low
        # a = np.random.normal(np.dot(self.__mu.T, observation), self.__sigma ** 2, 1)
        a = np.random.randn() * self.__sigma + np.dot(self.__mu.T, observation)
        a = min(a, max_a)
        a = max(a, min_a)

        return a

    def learnMC(self, L=100, M=100, T=600, ALPHA=0.3, GAMMA=0.1):
        # 政策反復

        for l in range(L):
            print("policy "+str(l))
            print(self.__mu, self.__sigma)
            drs = np.zeros(M)
            der = np.zeros([M,self.__N]) #パラメータ
            goal_count = 0
            for m in range(M):
                observation = env.reset()
                for t in range(T):
                    if m == M - 1 and l % 10 == 0:
                        env.render()

                    action = [0]
                    # 行動決定
                    action[0] = self.__get_action(observation)

                    # 行動
                    observation, reward, done, info = env.step(action)
                    observation = np.reshape(observation, (2,))
                    # μ計算
                    der[m][0:self.__N-1] += (action[0] - np.dot(self.__mu.T, observation)*observation)/(self.__sigma**2)
                    # σ計算
                    der[m][self.__N-1] += ((action[0] - np.dot(self.__mu.T, observation))**2 -self.__sigma**2) / (self.__sigma**3)
                    # 割引報酬和計算
                    drs[m] += GAMMA**(t) * reward

                    if done:
                        if t < T-2: # goal
                            print("Episode %d finished after {} timesteps".format(t) % m)
                            goal_count += 1
                        break
            print(str(goal_count) + "/" + str(M))

            # baseline
            baseLine = np.dot(drs,np.diag(np.dot(der,der.T)))/np.trace(np.dot(der,der.T))

            # 勾配推定
            der_J = (np.dot((drs - baseLine),der)) / M
            print(der_J)
            # モデルパラメータの推定
            self.__mu += ALPHA * der_J[0:self.__N-1]
            self.__sigma += ALPHA * der_J[self.__N-1]


if __name__ == '__main__':
    mc = MountainCar2()
    mc.learnMC(M=100, T=1000, ALPHA=0.1, GAMMA=1)