# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from mcts_pure import MCT_Pure_Player
from tensorboardX import SummaryWriter


class TrainPipeline:
    def __init__(self, model_path=None, **kwargs):
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.game = GameServer()
        # training params
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1  # 一次增加数据下多少次棋
        self.check_freq = 50
        self.game_batch_num = 1500  # 总共下多少次棋来训练
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        self.policy_value_net = PolicyNet(self.game)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        self.writer = SummaryWriter(log_dir='../data/record')

        self.bestModelPath = kwargs.get('bestPath', '../data/bestModel')
        self.checkPointPath = kwargs.get('checkPath', '../data/checkpoint')

    # 数据增强 因为盘面旋转相等
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    # 自己下n次棋 并且通过数据增强获取(s,a,v)
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    # 随机选取sample进行训练
    def policy_update(self):
        """update the policy-value net"""
        loss, entropy = self.policy_value_net.train(self.data_buffer, self.batch_size)
        return loss, entropy

    def evaluate_with_best(self, iteration, n_games=100):
        """
            Evaluate the trained policy by playing against the best past MCTS player
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        policy_value_net = PolicyNet(self.game).load_checkpoint(self.bestModelPath)
        best_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.play_a_game(current_mcts_player,
                                           best_mcts_player,
                                           start_player=i % 2,
                                           shown=False)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[-1], win_cnt[0]))

        self.writer.add_scalar('win_rate/p1 vs p2',
                               win_ratio, iteration)
        self.writer.add_scalar('win_rate/draws', win_cnt[0] / n_games, iteration)

        return win_ratio

    def evaluate_with_pure(self, iteration, n_games=100):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCT_Pure_Player(c_puct=5,
                                           n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.play_a_game(current_mcts_player,
                                           pure_mcts_player,
                                           start_player=i % 2,
                                           shown=False)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[-1], win_cnt[0]))

        self.writer.add_scalar('win_rate/p1 vs p2',
                               win_ratio, iteration)
        self.writer.add_scalar('win_rate/draws', win_cnt[0] / n_games, iteration)

        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Game i:{i + 1}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play game: {}".format(i + 1))
                    if i < 2 * self.check_freq:
                        win_ratio = self.evaluate_with_pure(i + 1)
                    else:
                        win_ratio = self.evaluate_with_best(i+1)
                    self.policy_value_net.save_checkpoint(self.checkPointPath)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_checkpoint(self.bestModelPath)
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
