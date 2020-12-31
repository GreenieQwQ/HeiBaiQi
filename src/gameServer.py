# -*- coding: utf-8 -*-

import time
from board import *
from utils import *
from progress.bar import Bar

# 可视化
class GameServer:
    def __init__(self, player1, player2, **kwargs):
        self.board = Board()
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        self.player1 = player1
        self.player2 = player2
        self.players = {p1: player1, p2: player2}

    # 将棋盘打印
    def graphic(self):
        self.board.print_board()

    # 开始游戏
    def play_a_game(self, start_player=0, shown=True):
        if start_player not in [0, 1]:
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        if shown:
            self.graphic()

        while True:
            end, winner = self.board.game_end()
            if end:
                if shown:
                    if winner is not None:
                        print("Game end. Winner is", self.players[winner])
                    else:
                        print("Game end. Tie")
                return winner

            current_player = self.board.get_current_player()
            player_in_turn = self.players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if shown:
                self.graphic()


    # 开始多局游戏——player1 开始 num / 2局游戏 player2 开始num / 2局游戏
    # 输出：oneWon——player1赢、twoWon——player2赢、draw——平局的个数
    def play_games(self, num, shown=False):
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            self.board.reset_board()
            gameResult = self.play_a_game(shown=shown)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100 * (oneWon + 0.5 * draws) / (oneWon + twoWon + draws)))
            bar.next()

        for _ in range(num):
            gameResult = self.play_a_game(shown=shown)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100 * (oneWon + 0.5 * draws) / (oneWon + twoWon + draws)))
            bar.next()

        bar.update()
        bar.finish()

        return oneWon, twoWon, draws