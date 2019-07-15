from common import *

Zero = torch.tensor(0.)
maxLen = 5

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = opt.device
        self.dtype = opt.dtype
        self.edim = opt.edim
        self.dropout = nn.Dropout(opt.dropout)
        self.to(dtype=opt.dtype, device=opt.device)
        self.f0 = nn.Linear(1, opt.edim, bias=True)
        self.act0 = nn.LeakyReLU(.1)
        self.norm = nn.BatchNorm1d(opt.edim * maxLen, affine=False)
        self.f1 = nn.Linear(opt.edim * maxLen, 1, bias=True)
        self.act1 = torch.tanh

    def forward(self, x, mask, *_):
        bsz, l = x.shape
        mask = mask.to(self.dtype)
        e = self.dropout(x).view(bsz, l, 1)
        x1 = self.act0(self.f0(e)) * mask.view(bsz, l, 1)
        x2 = self.norm(x1.view(bsz, -1)).view(bsz, l, -1) * mask.view(bsz, l, 1)
        return self.act1(self.f1(x2.view(bsz, -1)).squeeze(-1)), Zero, x1

predict = lambda x: x
