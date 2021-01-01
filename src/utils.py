import torch.utils.data as D
import numpy as np
import torch

class AverageMeter:
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SAV_Dataset(D.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # 功能：随机重排数据集
    def shuffle(self):
        np.random.shuffle(self.data)


class SAV_DataLoader(D.DataLoader):
    def __init__(self, dataset, device="cpu", **kwargs):
        super(SAV_DataLoader, self).__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.device = device

    # 读取batch的时候将数据转化为tensor
    def collate_fn(self, batches):
        states, target_pis, target_vs = [t[0] for t in batches], [t[1] for t in batches], [t[2] for t in batches]

        # prepare
        states, target_pis, target_vs = torch.FloatTensor(states).to(self.device), \
                                        torch.FloatTensor(target_pis).to(self.device), \
                                        torch.FloatTensor([target_vs]).to(self.device)

        return states, target_pis, target_vs

    # shuffle 数据
    def shuffle(self):
        self.dataset.shuffle()


if __name__ == "__main__":
    pass