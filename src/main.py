import pyximport; pyximport.install()
import argparse
from gameServer import GameServer
from NetWrapper import PolicyNet
from mct_player import MCTSPlayer
from mcts_pure import MCT_Pure_Player
import torch
import os
from NetWrapper import PolicyNet
from players import *
#from MCTS import cython_MCTS
from tqdm import trange

def run(**kwargs):
    # model_path = '../../fast-alphazero-general/checkpoint'
    # model_path = '../data/model_01_04_21_22_03_52/checkpoint_epoch_100'
    # model_path = '../data/model_01_03_21_22_21_34/checkpoint'
    model_path = '../data/test'
    MCT_path = '../data/test/mcts111.pkl'
    try:
        game = GameServer()
        policy_value_net1 = PolicyNet(game).load_checkpoint(model_path, 'iteration-0121.pkl')
        policy_value_net2 = PolicyNet(game).load_checkpoint(model_path, 'iteration-0121.pkl')
        human1 = RandomPlayer()
        h1temp = kwargs.get("t1", 0.1)
        h2temp = kwargs.get("t2", 0.1)
        print(f"H1temp: {h1temp}")
        print(f"H2temp: {h2temp}")
        human2 = MCTSPlayer(game, policy_value_net1)
        if os.path.isfile(MCT_path):
            print("Loading...model")
            human1 = torch.load(MCT_path)
            print("Loading complete")
        else:
            human1 = MCTSPlayer(game, policy_value_net2)

        human1 = MCTSPlayer(policy_value_net2.policy_value_fn, n_playout=100)
        human2 = MCT_Pure_Player(c_puct=5,
                                 n_playout=1000)
        human2 = RandomPlayer()

        # policy_value_net = PolicyNet(game).load_checkpoint(model_path)

        # human2 = MCTSPlayer(c_puct=5,
        #                     n_playout=400,
        #                     policy_value_function=policy_value_net.policy_value_fn)


        # human1 = MCT_Pure_Player(c_puct=5,  n_playout=1000)
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
        # one, two, draw = 0,0,0
        # for i in trange(10, ncols=80):
        #     # human1 = RandomPlayer()
        #     # human2 = torch.load(MCT_path)
        #     human2 = MCT_Pure_Player(c_puct=5,
        #                              n_playout=1000)
        #     winner = game.play_a_game(human1, human2, shown=False)
        #     if winner == -1:
        #         one += 1
        #     elif winner == 1:
        #         two += 1
        #     else:
        #         draw += 1
        # for i in trange(10, ncols=80):
        #     #human1 = RandomPlayer()
        #     #human2 = torch.load(MCT_path)
        #     human2 = MCT_Pure_Player(c_puct=5,
        #                              n_playout=1000)
        #     winner = game.play_a_game(human2, human1, shown=False)
        #     if winner == 1:
        #         one += 1
        #     elif winner == -1:
        #         two += 1
        #     else:
        #         draw += 1
        # print(one, two, draw)

        print(game.play_games(human1, human2, num=10, shown=False))
        # torch.save(human1, MCT_path)

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
