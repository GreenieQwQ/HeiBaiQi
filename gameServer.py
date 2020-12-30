from board import *

class BasePlayer:
    # 由server赋予index
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        return board.possible_moves()[0]

    def statistics(self):
        pass

# 可视化
class Game:
    def __init__(self, **kwargs):
        self.board = Othello()

    # 将棋盘打印
    def graphic(self):
        self.board.print_board()

    # 开始游戏
    def start_play(self, player1: BasePlayer, player2: BasePlayer, start_player=0, shown=True):
        self.p1, self.p2 = player1, player2
        if start_player not in [0, 1]:
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if shown:
            self.graphic()

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.play_move(move)
            if shown:
                self.graphic()
            end, winner = self.board.game_end()
            if end:
                if shown:
                    if winner is not None:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

