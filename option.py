from common import *
from model import vocab
option = dict(edim=16, epochs=3, maxgrad=1., sdt=1e-2, sdt_decay_step=1, batchsize=8, vocabsize=vocab, fp16=2)
option['loss'] = lambda opt, model, y, out, *_: F.cross_entropy(out.transpose(-1, -2), y, reduction='none')
option['criterion'] = lambda y, out, mask, *_: (out[:,:,1:vocab].max(-1)[1] + 1).ne(y).float() * mask.float()
try:
    from qhoptim.pyt import QHAdam
    option['newOptimizer'] = lambda opt, params, _: QHAdam(params, lr=opt.learningrate, nus=(.7, .8), betas=(0.995, 0.999))
except ImportError: pass
