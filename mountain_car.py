import gym
import numpy as np
import math
import random

# 山登りsample
env = gym.make('MountainCar-v0')

def basis_func(state, action):
    SIGMA = 0.5
    c = np.array([[-1.2, -1.5],[-1.2, -0.5],[-1.2, 0.5],[-1.2, 1.5],
                 [-0.35, -1.5],[-0.35, -0.5],[-0.35, 0.5],[-0.35, 1.5],
                 [0.5, -1.5],[0.5, -0.5],[0.5, 0.5],[0.5, 1.5]])
    phi = []
    for i in range(3):
        for j in range(len(c)):
            if action == i:
                I = 1
            else:
                I = 0
            phi.append(I*math.exp(-1*((np.linalg.norm(state-c[j])**2)/(2*(SIGMA**2)))))
    return np.array(phi)

def choose(probabilities):
    candidates = [0,1,2]
    probabilities = [sum(probabilities[:x+1]) for x in range(len(probabilities))]
    if probabilities[-1] > 1.0:
        #確率の合計が100%を超えていた場合は100％になるように調整する
        probabilities = [x/probabilities[-1] for x in probabilities]
    rand = random.random()
    for candidate, probability in zip(candidates, probabilities):
        if rand < probability:
            return candidate
    #どれにも当てはまらなかった場合はNoneを返す
    return None

def max_Q(s, theta):
    Q = []
    for a in range(3):
        Q.append(np.dot(theta.T, basis_func(s, a)))

    indexes = [i for i, x in enumerate(Q) if x == max(Q)]
    return random.choice(indexes)

def learn_MC(L=100, M=10, T=200, EPSIL=0.1,GAMMA=0.1, options=2):
    B = 36
    # パラメータ
    theta = np.random.rand(B)

    # 政策反復
    for l in range(L):
        # 標本抽出
        x = [] # M*T*B
        r = [] # M*T
        print("policy")
        for m in range(M): # エピソード
            observation = env.reset()
            for t in range(T): # ステップ
                # env.render()

                # 政策改善方法
                action_p = [0,0,0]
                if options == 1:  # greedy
                    a = max_Q(observation, theta)
                    action_p[a] = 1
                elif options == 2:  # ε-greedy
                    a = max_Q(observation, theta)
                    action_p = [(EPSIL / len(action_p)) for i in range(len(action_p))]
                    action_p[a] = 1 - EPSIL + (EPSIL / len(action_p))

                action = choose(action_p)

                if t >= 1:
                    x.append(basis_func(pre_observation, pre_action)- (GAMMA*basis_func(observation, action)))
                    r.append(pre_reward)

                pre_observation = observation
                pre_action = action

                # 行動
                observation, reward, done, info = env.step(action)

                pre_reward = reward
                if done:  # goal
                    print("Episode %d finished after {} timesteps".format(m) % t)
                    break

        X = np.array(x).reshape((M,-1))
        R = np.array(r).reshape((M,-1))

        theta = np.dot(np.dot(np.linalg.pinv((np.dot(X.T,X))),X.T), R)

if __name__ == '__main__':
    learn_MC(M=10)