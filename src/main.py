from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from mcts_pure import MCT_Pure_Player
from NetWrapper import PolicyNet
from players import *


def run():
    model_path = '../data/model_01_03_21_14_58_17/checkpoint'
    try:
        human1 = RandomPlayer()
        human2 = Human()
        # human2 = MCT_Pure_Player(c_puct=5,
        #                          n_playout=1000)

        game = GameServer()

        policy_value_net = PolicyNet(game).load_checkpoint(model_path)
        human2 = MCTSPlayer(c_puct=5,
                            n_playout=400,
                            policy_value_function=policy_value_net.policy_value_fn)
        # human2 = MCT_Pure_Player(c_puct=5,  n_playout=1000)

        # policy_value_net = PolicyNet(game).load_checkpoint(model_path)
        # human2 = MCTSPlayer(c_puct=5,
        #                     n_playout=400,
        #                     policy_value_function=policy_value_net.policy_value_fn)
        #human2 = MCT_Pure_Player(c_puct=5,  n_playout=400)

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
        game.play_a_game(human1, human2, start_player=1)
        #print(game.play_games(human1, human2, num=20, shown=False))
    except KeyboardInterrupt:
        print('\n\r\n\rQuit.')


if __name__ == '__main__':
    run()
