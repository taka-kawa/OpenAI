import copy, sys
import numpy as np
from collections import deque
import gym

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class Neuralnet(Chain):
    """
    ニューラルネットワーク部分
    """
    def __init__(self, n_in, n_out, unit=100):
        super(Neuralnet, self).__init__(
            L1 = L.Linear(n_in, unit),
            L2 = L.Linear(unit, unit),
            L3 = L.Linear(unit, unit),
            Q_value = L.Linear(unit, n_out, initialW=np.zeros((n_out, unit), dtype=np.float32))
        )
    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = self.Q_value(h)
        return h

class Agent():
    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        self.n_act = n_act
        self.model = Neuralnet(n_st, n_act)
        """
        浅いコピー(shallow copy)は新たな複合オブジェクトを作成し、その後(可能な限り)元のオブジェクト中に見つかったオブジェクトに対する参照を挿入する。
        深いコピー(deep copy)は新たな複合オブジェクトを作成し、その後元のオブジェクト中に見つかったオブジェクトのコピーを挿入する。
        """
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque() # deque()はpopとappendを左右どちらからでもできる
        self.loss = 0
        self.step = 0
        self.gamma = 0.99  # 割引率
        self.mem_size = 1000  # Experience Replayのために覚えておく経験の数
        self.batch_size = 100  # Experience Replayの際のミニバッチの大きさ
        self.train_freq = 10  # ニューラルネットワークの学習間隔
        self.target_update_freq = 20  # ターゲットネットワークの同期間隔
        # ε-greedy
        self.epsilon = 1  # εの初期値
        self.epsilon_decay = 0.005  # εの減衰値
        self.epsilon_min = 0  # εの最小値
        self.exploration = 1000  # εを減衰し始めるまでのステップ数(今回はメモリーが貯まるまで)

    # 経験の蓄積
    def stock_experience(self, st, act, r, st_dash, ep_end):
        """
        5つの要素を経験としてタプルにしてmemoryに保存している。最初に定義したmemoryサイズを超えると先に入れたものからトコロテン式に捨てられる形になっている。
        :param st: 状態
        :param act: 行動
        :param r: 報酬
        :param st_dash: 次の状態
        :param ep_end: エピソード終了の有無
        :return: 
        """
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    # データ整形
    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(range(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start + self.batch_size]
            batch = mem[index]
            st, act, r, st_d, ep_end = self.parse_batch(batch)
            self.model.zerograds()
            loss = self.forward(st, act, r, st_d, ep_end)
            loss.backward()
            self.optimizer.update()

    def forward(self, st, act, r, st_dash, ep_end):
        """
        ニューラルネットワークを使ったQ関数の更新部分。
        次の状態(st_dash)のQ値の最大値を計算する部分ではコピーしたQ関数(self.target_model.Q_func)を使うところが重要
        
        :param st: 状態
        :param act: 行動
        :param r: 報酬
        :param st_dash: 次の状態 
        :param ep_end: ステップの終了
        :return: 
        """
        s = Variable(st)
        s_dash = Variable(st_dash)
        # 現在の状態のQ関数
        Q = self.model.Q_func(s)
        # 次の状態の教師Q関数(次の状態はmainでもらう)
        tmp = self.target_model.Q_func(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in range(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        # 現在の状態のQ関数の出力と次状態の教師Q関数の誤差
        loss = F.mean_squared_error(Q, Variable(target))
        return loss

    def get_action(self, st):
        """
        学習したQ関数に従って入力された状態の時に取るべき行動を返す部分です。行動選択の手法はε-greedyを使っている。
        :param st: 
        :return: 
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act)
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8)

    # 学習を進める
    """
    メモリが十分に溜まったら学習を進める部分
    毎回stepを刻んで、一定周期でtarget用のQ関数を同期
    """
    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        # メモリが指定メモリ数を超え出したらやっと更新していく
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

def main(env_name, epoch=1000, T=200):
    env = gym.make(env_name)
    # view_path = "./video/" + env_name
    seed = 0

    n_st = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = range(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = Agent(n_st, n_act, seed)

    # env.monitor.start(view_path, video_callable=None, force=True, seed=seed)
    for i_episode in range(epoch):
        print("episode", i_episode)
        observation = env.reset()
        for t in range(T):
            if i_episode % 1 == 0:
                env.render()
            # 現在の行動保持
            state = observation.astype(np.float32).reshape((1, n_st))
            # Q関数に従って行動選択
            act_i = agent.get_action(state)
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            # 前の状態
            state_dash = observation.astype(np.float32).reshape((1, n_st))
            # 経験を溜める(メモリが溜まっていく)
            agent.stock_experience(state, act_i, reward, state_dash, ep_end)
            # 溜めるメモリに達すれば更新
            agent.train()
            if ep_end:
                break
    env.monitor.close()

if __name__ == "__main__":
    main('MountainCar-v0')