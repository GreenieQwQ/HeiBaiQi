import numpy as np


class Board:
    def __init__(self, width=8):
        self.width = width
        self.reset_board(self.width)

    def getBoardSize(self):
        return self.width, self.width

    # +1 !!! 黑白棋不能下可以pass！！！
    def getActionSize(self):
        # return number of actions
        return self.width * self.width + 1

    def getState(self):
        return np.copy(self.board)

    def reset_board(self, width=8, start_player=0):
        self.board = np.zeros((width, width), dtype=np.int)
        self.players = [-1, 1]
        self.current_player = self.players[start_player]  # start player
        self.board[3, 3] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        self.board[4, 4] = 1
        self.availables = self.get_valid_moves()

    def get_current_player(self):
        return self.current_player

    def get_opponent(self):
        return (
            self.players[0]
            if self.current_player == self.players[1]
            else self.players[1]
        )

    def do_move(self, move):
        x, y = Board.move_to_location(move)
        side = self.current_player
        if move not in self.possible_moves():
            raise ValueError("Invalid move")
        if x == -1 and y == -1:
            # print(str(self.current_player) + "pass")
            self.current_player = self.get_opponent()  # 换边
            self.availables = self.get_valid_moves()  # 注意要在换边之后
            return
        self.board[x, y] = side
        self.flip(x, y, side)
        self.current_player = self.get_opponent()  # 换边
        self.availables = self.get_valid_moves()    # 注意要在换边之后

    def game_end(self):
        is_over = self.is_game_over()
        if not is_over:
            return is_over, 0
        return is_over, self.get_winner()

    # 要黑白两方都没有棋下才行！
    def is_game_over(self):
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == 0 and (self.valid_flip(i, j, -1) or self.valid_flip(i, j, 1)):
                    return False
        return True

    def get_winner(self):
        t = np.sum(self.board)
        if t > 0:
            return 1
        if t < 0:
            return -1
        return 0

    def get_valid_moves(self):
        side = self.get_current_player()
        moves = []
        for i in range(self.width):
            for j in range(self.width):
                if self.board[i, j] == 0 and self.valid_flip(i, j, side):
                    moves.append(Board.location_to_move((i, j)))
        return moves if moves else [self.width * self.width]

    def possible_moves(self):
        return self.availables

    def valid_flip(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if self.valid_ray(x, y, side, dx, dy):
                    return True
        return False

    def valid_ray(self, x, y, side, dx, dy):
        tx = x + 2 * dx
        if tx < 0 or tx > 7:
            return False
        ty = y + 2 * dy
        if ty < 0 or ty > 7:
            return False
        if self.board[x + dx, y + dy] != -1 * side:
            return False
        while self.board[tx, ty] != side:
            if self.board[tx, ty] == 0:
                return False
            tx += dx
            ty += dy
            if tx < 0 or tx > 7:
                return False
            if ty < 0 or ty > 7:
                return False
        return True

    def flip(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if self.valid_ray(x, y, side, dx, dy):
                    self.flip_ray(x, y, side, dx, dy)

    def flip_ray(self, x, y, side, dx, dy):
        tx = x + dx
        ty = y + dy
        while self.board[tx, ty] != side:
            self.board[tx, ty] = side
            tx += dx
            ty += dy

    def print_board(self):
        print()
        print(" " * 2, end="")
        for x in range(self.width):
            print("{0:4}".format(x), end='')
        print()
        for i in range(self.width):
            print("{0:4d}".format(i), end='')
            for j in range(self.width):
                p = Board.piece_map(self.board[i, j])
                print(p.center(4), end='')
            print()

    @staticmethod
    def piece_map(x):
        return {
            1: 'W',
            -1: 'B',
            0: '-',
        }[x]

    @staticmethod
    def location_to_move(location, width=8):
        if location == (-1, -1):
            return 64
        return location[0] + location[1] * width

    move_count = 65

    @staticmethod
    def move_to_location(move, width=8):
        if move == 64:
            return (-1, -1)
        x = move % width
        y = move // width
        return (x, y)

    @staticmethod
    def state_id(board):
        x = np.add(board, 1).flatten()
        id = 0
        mult = 1
        for t in x:
            id += mult * int(t)
            mult *= 3
        return id
