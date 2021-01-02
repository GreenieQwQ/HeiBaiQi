from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from players import *

def run():
    model_path = '../data/bestModel'
    try:
        human1 = Human()
        human2 = RandomPlayer()
        game = GameServer()

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyNet(game).load_checkpoint(model_path)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        #mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3


        # set start_player=0 for human first
        # game.play_a_game(human1, human2, start_player=1)
        game.play_games(mcts_player, mcts_player, num=100)
    except KeyboardInterrupt:
        print('\n\r\n\rQuit.')


if __name__ == '__main__':
    run()