import pyximport; pyximport.install()
import argparse
from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from mcts_pure import MCT_Pure_Player
from NetWrapper import PolicyNet
from players import *
from MCTS import cython_MCTS

def run(**kwargs):
    # model_path = '../../fast-alphazero-general/checkpoint'
    # model_path = '../data/model_01_04_21_22_03_52/checkpoint_epoch_100'
    # model_path = '../data/model_01_03_21_22_21_34/checkpoint'
    model_path = '../data/test'
    try:
        game = GameServer()
        policy_value_net1 = PolicyNet(game).load_checkpoint(model_path, 'iteration-0163.pkl')
        policy_value_net2 = PolicyNet(game).load_checkpoint(model_path, 'iteration-0163.pkl')
        human1 = RandomPlayer()
        h1temp = kwargs.get("t1", 0.1)
        h2temp = kwargs.get("t2", 0.1)
        print(f"H1temp: {h1temp}")
        print(f"H2temp: {h2temp}")
        # human1 = cython_MCTS(game, policy_value_net1, temp=h1temp, numMCTSSims=600)
        human2 = cython_MCTS(game, policy_value_net2, temp=h2temp, numMCTSSims=600)

        # human2 = MCT_Pure_Player(c_puct=5,
        #                          n_playout=1000)


        # policy_value_net = PolicyNet(game).load_checkpoint(model_path)

        # human2 = MCTSPlayer(c_puct=5,
        #                     n_playout=400,
        #                     policy_value_function=policy_value_net.policy_value_fn)
        # human2 = MCT_Pure_Player(c_puct=5,  n_playout=400)
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyNet(game).load_checkpoint(model_path)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

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
        # mcts_player = MCT_Pure_Player(c_puct=5, n_playout=400)

        # human player, input your move in the format: 2,3

        # set start_player=0 for human first
        # game.play_a_game(human1, human2, start_player=1)
        print(game.play_games(human1, human2, num=20, shown=True))
    except KeyboardInterrupt:
        print('\n\r\n\rQuit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaZero Othello')
    parser.add_argument('--t1', type=float, default=0.1,
                        help="p1's temp")
    parser.add_argument('--t2', type=float, default=0.1,
                        help="p2's temp")
    args = parser.parse_args()
    run(t1=args.t1, t2=args.t2)
