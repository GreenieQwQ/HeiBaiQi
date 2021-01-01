from OthelloNet import OthelloNet
import torch.optim as optim
import torch
from time import time
import os
import numpy as np
from utils import AverageMeter
from progress.bar import Bar
from board import Board


class Args:
    lr = 0.01
    dropout = 0.3
    cuda = torch.cuda.is_available()
    num_channels = 512
    depth = 10


args = Args
# Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class PolicyNet:
    def __init__(self, game):
        self.nnet = OthelloNet(game, num_channels=args.num_channels, dropout=args.dropout)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(
            self.nnet.parameters(), lr=args.lr)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #    self.optimizer, milestones=[200,400], gamma=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, cooldown=10)

    # batches = [(s, pi, v)]
    # epoches: 在batch上训练的次数
    def train(self, batches, train_steps=5):
        # 随机打乱batch
        np.shuffle(batches)
        self.nnet.train()

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        end = time()

        # print(f'Current LR: {self.scheduler.get_lr()[0]}')
        bar = Bar(f'Training Net', max=train_steps)
        current_step = 0
        while current_step < train_steps:
            for batch_idx, batch in enumerate(batches):
                if current_step == train_steps:
                    break
                current_step += 1
                boards, target_pis, target_vs = batch

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                # plot progress
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
        self.scheduler.step(pi_losses.avg + v_losses.avg)
        bar.finish()
        print()

        return pi_losses.avg, v_losses.avg

    def policy_value(self, state_batch):
        """
            input: a batch of states
            output: a batch of action probabilities and state values
        """
        with torch.no_grad():
            self.nnet.eval()
            log_act_probs, value = self.nnet(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board: Board):
        """
            input: board
            output: a list of (action, probability) tuples for each available
            action and the score of the board state
        """
        # preparing input
        input_board = board.getState()
        input_board = torch.FloatTensor(input_board.astype(np.float64))
        with torch.no_grad():
            input_board = input_board.view(1, self.board_x, self.board_y)
            self.nnet.eval()
            pi, v = self.nnet(input_board)

        # 过滤非法动作
        act_probs = torch.exp(pi).data.cpu().numpy()[0]
        value = v.data.cpu().numpy()[0]
        availables = board.possible_moves()
        return zip(availables, act_probs[availables]), value

    def process(self, batch):
        if args.cuda:
            batch = batch.cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch)
            return torch.exp(pi), v

    # 交叉熵 output已经过log_softmax层
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    # 均方误差
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='../data/checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='../data/checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'opt_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt_state'])
        if 'sch_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['sch_state'])
