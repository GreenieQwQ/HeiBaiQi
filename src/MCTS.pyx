# cython: language_level=3
# cython: linetrace=True
# cython: profile=True
# cython: binding=True

import math
import numpy as np
from players import BasePlayer

EPS = 1e-8

# 相比原蒙特卡罗方法还使用了置换表加速搜索？
class cython_MCTS(BasePlayer):
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, **kwargs):
        self.game = game
        self.nnet = nnet
        self.numMCTSSims = kwargs.get('numMCTSSims', 50)
        self.cpuct = kwargs.get('cpuct', 3)
        self.reset()
        self.temp = kwargs.get('temp', 0.1)
        self.tempThreshold = kwargs.get('tempThreshold', 10)

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.mode = 'leaf'
        self.path = []
        self.v = 0

    def get_action(self, board):
        inputBoard = board.getCanonicalForm()
        policy = self.getActionProb(inputBoard, temp=self.temp)
        return np.random.choice(len(policy), p=policy)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        try:
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
        except OverflowError as err:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        # 若为终止结点则直接返回value即可
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # 未遍历过 未知使用Net得到的先验概率——MSTS的叶节点
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # 网络将有效动作结点mask了 可能发生了过拟合
                # 将所有有效动作节点重新均分
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # 内部节点
        valids = self.Vs[s]     # 合法动作
        cur_best = -float('inf')
        best_act = -1

        # 选择最大UCT来进行遍历
        # 遍历全部动作 过滤掉非法动作
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:   # 遍历全部动作 过滤掉非法动作
                if (s, a) in self.Qsa:  # 平均行动价值 选择最大Q+U分支
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * \
                        self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:    # 记录最大Q+U分支
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
