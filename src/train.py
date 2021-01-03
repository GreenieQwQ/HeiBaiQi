# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import os
import numpy as np
import torch
import time
import argparse
from collections import defaultdict, deque
from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from mcts_pure import MCT_Pure_Player
from tensorboardX import SummaryWriter
from tqdm import trange
from players import RandomPlayer


class TrainPipeline:
    def __init__(self, **kwargs):
        # params of the board and the game
        self.game = GameServer()
        self.board_width, self.board_height = self.game.getBoardSize()
        # training params
        self.temp = kwargs.get('temp', 1.0)  # the temperature param
        self.n_playout = kwargs.get('n_playout', 400)  # num of simulations for each move
        self.c_puct = kwargs.get('c_puct', 5)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.batch_size = kwargs.get('batch_size', 512)  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_per_iter = kwargs.get('play_per_iter', 1)  # 一次增加数据下多少次棋
        self.check_freq = kwargs.get('check_freq', 100)  # 多少次迭代evaluate
        self.epoch_num = kwargs.get('epoch_num', 6000)  # 训练的迭代次数
        self.train_steps = kwargs.get('train_steps', 500)   # 一个epoch在batch上迭代的次数

        self.beaten_pure_mct = False
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        self.lr = kwargs.get('lr', 0.005)
        self.dropout = kwargs.get('dropout', 0.3)

        self.policy_value_net = PolicyNet(self.game, lr=self.lr, dropout=self.dropout)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)

        self.id = 'model_' + time.strftime("%D_%H_%M_%S", time.localtime()).replace("/", "_")
        self.train_dir = '../data/' + self.id
        self.log_dir = kwargs.get('log_dir', self.train_dir + '/record')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.bestModelPath = kwargs.get('bestPath', self.train_dir + '/bestModel')
        self.checkPointPath = kwargs.get('checkPath', self.train_dir + '/checkpoint')

        # 加载模型和训练状态
        loadPath = kwargs.get('loadPath', None)
        if loadPath:
            self.policy_value_net.load_checkpoint(loadPath)
            self.loadTrainState(loadPath)

    # 记录训练的state
    def saveTrainState(self, folder, filename='state.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save({
            'best_win_ratio': self.best_win_ratio,
            'beaten_pure_mct': self.beaten_pure_mct,
            'pure_mcts_playout_num': self.pure_mcts_playout_num,
            'databuffer': self.data_buffer
        }, filepath)
        return self

    def loadTrainState(self, folder, filename='state.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            print("No state in path {}".format(filepath))
            return self
        state_dict = torch.load(filepath)
        self.best_win_ratio = state_dict['best_win_ratio']
        self.beaten_pure_mct = state_dict['beaten_pure_mct']
        self.pure_mcts_playout_num = state_dict['pure_mcts_playout_num']
        return self

    # 数据增强 因为盘面旋转相等
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...] (s, a, v)
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            # 注意：先提取出第65个元素
            stop_prob = mcts_porb[-1]
            square_prob = mcts_porb[:-1]
            for i in [1, 2, 3, 4]:
                # print(state)
                # print(mcts_porb)
                # print(winner)
                # rotate counterclockwise
                equi_state = np.rot90(state, i)
                equi_mcts_prob = np.rot90(np.flipud(
                    square_prob.reshape(self.board_height, self.board_width)), i)
                final_equi_mcts_prob = np.flipud(equi_mcts_prob).flatten()  # 为了加上第65个元素单独列出
                extend_data.append((equi_state,
                                    np.append(final_equi_mcts_prob, stop_prob),
                                    winner))
                # flip horizontally
                equi_state = np.fliplr(equi_state)
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                final_equi_mcts_prob = np.flipud(equi_mcts_prob).flatten()  # 为了加上第65个元素单独列出
                extend_data.append((equi_state,
                                    np.append(final_equi_mcts_prob, stop_prob),
                                    winner))
        return extend_data

    # 自己下n次棋 并且通过数据增强获取(s,a,v)
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in trange(n_games, ncols=80):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp, shown=False)
            play_data = list(play_data)[:]
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            # 以上为采样过程

    # 随机选取sample进行训练
    def policy_update(self):
        loss, entropy = self.policy_value_net.train(self.data_buffer, self.batch_size)
        return loss, entropy

    # 和自己比较
    def evaluate_with_best(self, iteration, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        policy_value_net = PolicyNet(self.game).load_checkpoint(self.bestModelPath)
        best_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        win_cnt[1], win_cnt[-1], win_cnt[0] = self.game.play_games(current_mcts_player, best_mcts_player, n_games)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("Self Play: num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[-1], win_cnt[0]))

        self.writer.add_scalar('win_rate_self/p1 vs p2',
                               win_ratio, iteration)
        self.writer.add_scalar('win_rate_self/draws', win_cnt[0] / n_games, iteration)

        return win_ratio

    # 和随机下棋的比较
    def evaluate_with_random(self, iteration, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        random_player = RandomPlayer()
        win_cnt = defaultdict(int)
        win_cnt[1], win_cnt[-1], win_cnt[0] = self.game.play_games(current_mcts_player, random_player, n_games)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("Random: num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[-1], win_cnt[0]))

        self.writer.add_scalar('win_rate_random/p1 vs p2',
                               win_ratio, iteration)
        self.writer.add_scalar('win_rate_random/draws', win_cnt[0] / n_games, iteration)

        return win_ratio

    # 和纯蒙特卡罗法比较
    def evaluate_with_pure(self, iteration, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCT_Pure_Player(c_puct=5,
                                           n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        win_cnt[1], win_cnt[-1], win_cnt[0] = self.game.play_games(current_mcts_player, pure_mcts_player, n_games)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("Pure MCT: num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[-1], win_cnt[0]))

        self.writer.add_scalar('win_rate_pure/p1 vs p2',
                               win_ratio, iteration)
        self.writer.add_scalar('win_rate_pure/draws', win_cnt[0] / n_games, iteration)

        return win_ratio

    # 运行pipeline
    def run(self):
        try:
            for i in range(self.epoch_num):
                self.collect_selfplay_data(self.play_per_iter)
                iteration = i + 1  # 迭代的次数
                print(f"Game i:{iteration}")
                if len(self.data_buffer) > self.batch_size * 2:
                    l_pi, l_v = self.policy_update()
                    # 记录流程
                    self.writer.add_scalar('loss/policy', l_pi, iteration)
                    self.writer.add_scalar('loss/value', l_v, iteration)
                    self.writer.add_scalar('loss/total', l_pi + l_v, iteration)
                # check the performance of the current model,
                # and save the model params
                self.policy_value_net.save_checkpoint(self.checkPointPath)
                if (i + 1) % self.check_freq == 0:
                    print("current self-play game: {}".format(i + 1))
                    self.saveTrainState(self.checkPointPath)
                    # if ((i + 1) // self.check_freq) % 2 != 0:
                    if not self.beaten_pure_mct:  # 未完全胜利
                        win_ratio = self.evaluate_with_pure(i + 1)
                    else:  # 完全胜利纯蒙特卡洛 后续和自己比较
                        win_ratio = self.evaluate_with_best(i + 1)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 提升纯蒙特卡罗的强度
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                        elif self.pure_mcts_playout_num == 5000:
                            self.beaten_pure_mct = True     # 进无可进
                        # endif
                        # update the best_policy
                        self.policy_value_net.save_checkpoint(self.bestModelPath)
                        self.saveTrainState(self.bestModelPath)
                    # endif
                # endif
            # endfor
        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaZero Othello')
    parser.add_argument('--loadDir', type=str, default='../data/model_01_03_21_16_18_02/checkpoint',
                        help="checkpoint's directory")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--cf', type=int, default=150,
                        help='check freq')

    args = parser.parse_args()
    training_pipeline = TrainPipeline(loadPath=args.loadDir, lr=args.lr, check_freq=args.cf)
    training_pipeline.run()
