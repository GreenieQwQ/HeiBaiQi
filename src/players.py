import random

class BasePlayer:
    # 由server赋予index
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        return board.possible_moves()[0]

    def reset(self):
        pass

    def statistics(self):
        pass

class RandomPlayer(BasePlayer):
    def get_action(self, board):
        return random.choice(board.possible_moves())

    def __str__(self):
        return "RandomPlayer {}".format(self.player)

class Human(BasePlayer):

    def get_action(self, board):
        # 无可下的子时
        if not board.possible_moves():
            print("No move for you!")
            return (-1, -1)
        try:
            print("input -1 to Undo, -2 to save board")
            location = input("Your move: ")
            # location为int的列表[i, j]
            if isinstance(location, str):
                location = [int(n) for n in location.split(",")]
            if location[0] == -1:
                board.undo()
                move = None
            elif location[0] == -2:
                board.save_board()
                move = None
            else:
                move = board.location_to_move(location)
        except Exception:
            move = None

        if move is None or move not in board.possible_moves():
            print("invalid move")
            move = self.get_action(board)
        return move #move是(i, j)形式

    def __str__(self):
        return "Human {}".format(self.player)
