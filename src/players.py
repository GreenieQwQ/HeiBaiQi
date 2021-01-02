import random

class BasePlayer:
    # 由server赋予index
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        return board.possible_moves()[0]

    def statistics(self):
        pass

class RandomPlayer(BasePlayer):
    def get_action(self, board):
        try:
            return random.choice(board.possible_moves())
        except IndexError:
            pass

    def __str__(self):
        return "RandomPlayer {}".format(self.player)

class Human(BasePlayer):

    def get_action(self, board):
        # 无可下的子时
        if not board.possible_moves():
            print("No move for you!")
            return None
        try:
            location = input("Your move: ")
            # location为int的列表[i, j]
            if isinstance(location, str):
                location = [int(n) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = None

        if move is None or move not in board.possible_moves():
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)