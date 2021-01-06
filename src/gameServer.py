# -*- coding: utf-8 -*-

import time
from board import *
from utils import *
from progress.bar import Bar
from mct_player import MCTSPlayer


# 可视化
class GameServer:
    def __init__(self, **kwargs):
        self.board = Board()
        self.n = self.board.width

    def getBoardSize(self):
        return self.board.getBoardSize()

    def getActionSize(self):
        return self.board.getActionSize()

    # 将棋盘打印
    def graphic(self):
        self.board.print_board()

    def stringRepresentation(self, board):
        return board.tostring()

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board()
        b.initilize(board, player)
        is_end, winner = b.game_end()
        if not is_end:
            return 0
        else:
            return 1 if winner == 1 else -1

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board()
        b.initilize(board, player)
        legalMoves = b.get_valid_moves()
        for m in legalMoves:
            valids[m] = 1
        return np.array(valids)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.initilize(board, player)
        b.do_move(action)
        return (np.asarray(b.board), -player)

    # 开始游戏
    def play_a_game(self, player1, player2, start_player=0, shown=True):
        # 重启盘面
        self.board.reset_board(start_player=start_player)

        # 设置player
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

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
                        print("Game end. Winner is", winner)
                    else:
                        print("Game end. Tie")
                return winner

            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if shown:
                self.graphic()

    # 使用蒙特卡洛player自己和自己下棋 并且为训练记录(s, a, v)
    def start_self_play(self, player: MCTSPlayer, shown=False, temp=1e-3):
        self.board.reset_board()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=True)
            # store the data
            states.append(self.board.getState())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if shown:
                self.graphic()
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                v = np.zeros(len(current_players))
                if winner != 0:  # 平局
                    v[np.array(current_players) == winner] = 1.0
                    v[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, v)

    # 开始多局游戏——player1 开始 num / 2局游戏 player2 开始num / 2局游戏
    # 输出：oneWon——player1赢、twoWon——player2赢、draw——平局的个数
    def play_games(self, player1, player2, num=10, shown=False):
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        black = 0
        white = 0

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            self.board.reset_board()
            player1.reset()
            player2.reset()
            gameResult = self.play_a_game(player1, player2, shown=shown)
            if gameResult == -1:
                oneWon += 1
                black += 1
            elif gameResult == 1:
                twoWon += 1
                white += 1
            else:
                draws += 1
                white += 0.5
                black += 0.5
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100 * (oneWon + 0.5 * draws) / (oneWon + twoWon + draws)))
            bar.next()

        for _ in range(num):
            gameResult = self.play_a_game(player2, player1, shown=shown)
            if gameResult == 1:
                oneWon += 1
                white += 1
            elif gameResult == -1:
                twoWon += 1
                black += 1
            else:
                draws += 1
                white += 0.5
                black += 0.5
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

        print(f"Black: {black}")
        print(f"White: {white}")

        return oneWon, twoWon, draws
