from common import *
from torch.utils.data import Dataset, DataLoader

dataLength = {'train': 65536, 'val': 256, 'test': 256}

class Data(Dataset):
    def __init__(self, path):
        super(Data, self).__init__()
        l = dataLength[path]
        self.lens = torch.randint(4, (l,)) + 1
        self.mask = torch.zeros((l, 5), dtype=torch.uint8)
        for i in range(l):
            self.mask[i, :self.lens[i]].fill_(1)
        self.data = torch.rand((l, 5)) * self.mask.float()
        self.count = l
    def __len__(self):
        return self.count
    # input, label, length, mask
    def __getitem__(self, ind):
        x = self.data[ind]
        return x, x.sum(), self.lens[ind], self.mask[ind]

newLoader = lambda path, *args, **kwargs: DataLoader(Data(path), *args, **kwargs)
