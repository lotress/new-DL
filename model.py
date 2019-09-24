from common import *

Zero = torch.tensor(0.)

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = opt.device
        self.dtype = opt.dtype
        self.edim = opt.edim
        self.eps = 1e-4 if opt.fp16 else 1e-8
        vocabsize = opt.vocabsize
        self.embedding = nn.Embedding(vocabsize, opt.edim)
        self.dropout = nn.Dropout(opt.dropout)
        self.to(dtype=opt.dtype, device=opt.device)
        self.f0 = nn.Linear(opt.edim, opt.edim, bias=True)
        self.act0 = nn.LeakyReLU(.1)
        self.f1 = nn.Linear(opt.edim, vocabsize, bias=False)

    def forward(self, x, mask, *_):
        bsz, l = x.shape
        e = self.dropout(self.embedding(x))
        mask = mask.to(e.dtype).unsqueeze(-1)
        x1 = self.act0(self.f0(e)) * mask
        x2 = F.normalize(x1, dim=1, eps=self.eps) * mask
        return self.f1(x2), Zero.to(self.device), x1

with open('integrationTests.dict', encoding='utf-8') as fd:
    idict = [line.split('\t')[0] for line in fd if not line.startswith('__FP16_PAD_')]
    vocab = len(idict)
predict = lambda x, l: [' '.join(idict[i] for i in seq[:l[k]]) for k, seq in enumerate(x[:,:,1:vocab].max(-1)[1] + 1)]
