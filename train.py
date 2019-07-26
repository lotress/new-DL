import torch.optim as optim
import numpy as np
from common import *
from data import newLoader
from model import Model, predict
from option import option
if amp:
    from apex.optimizers import FusedAdam
torch.manual_seed(args.rank)
np.random.seed(args.rank)
getNelement = lambda model: sum(map(lambda p: p.nelement(), model.parameters()))
l1Reg = lambda acc, cur: acc + cur.abs().sum(dtype=torch.float)
l2Reg = lambda acc, cur: acc + (cur * cur).sum(dtype=torch.float)
nan = torch.tensor(float('nan'), device=opt.device)

opt.batchsize = 1
opt.epochs = 1
opt.maxgrad = 1. # max gradient
opt.dropout = 0
opt.sdt = 0.001 # initial learning rate
opt.sdt_decay_step = 10 # how often to reduce learning rate
opt.criterion = lambda y, out, mask: F.mse_loss(out, y) # criterion for evaluation
opt.loss = lambda opt, model, y, out, *args: F.mse_loss(out, y) # criterion for loss function
opt.newOptimizer = (lambda opt, params, _: FusedAdam(params, lr=opt.sdt)) if amp else lambda opt, params, eps: optim.Adam(params, lr=opt.sdt, amsgrad=True, eps=eps)
opt.writer = 0 # TensorBoard writer
opt.drawVars = 0
opt.reset_parameters = 0
opt.__dict__.update(option)

def initParameters(opt, model):
    for m in model.modules():
        if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.PReLU):
            nn.init.constant_(next(m.parameters()), 1)
        if opt.reset_parameters:
            opt.reset_parameters()
    if hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding):
        model.embedding.weight.data[2:] = torch.load(word2vecPath)

getParameters = (lambda opt, _: amp.master_params(opt.optimizer)) if amp else lambda _, model: model.parameters()
backward = lambda loss, _: loss.backward()
if torch.cuda.is_available():
    def backward(loss, opt):
        with amp.scale_loss(loss, opt.optimizer) as scaled_loss:
            scaled_loss.backward()

def trainStep(opt, model, x, y, length, mask):
    opt.optimizer.zero_grad()
    x = x.to(opt.device, non_blocking=True)
    mask = mask.to(opt.device, non_blocking=True)
    label = y.to(opt.device, dtype=torch.float, non_blocking=True)
    loss = opt.loss(opt, model, label, *model(x, mask))
    if torch.allclose(loss, nan, equal_nan=True):
        raise Exception('Loss returns NaN')
    backward(loss, opt)
    nn.utils.clip_grad_value_(getParameters(opt, model), opt.maxgrad)
    opt.optimizer.step()
    return float(loss)

def evaluateStep(opt, model, x, y, _, mask):
    mask = mask.to(opt.device, non_blocking=True)
    out, *others = model(x.to(opt.device, non_blocking=True), mask)
    pred = predict(out)
    if isinstance(pred, torch.Tensor):
        y = y.to(pred)
    missed = opt.criterion(y, pred, mask)
    return (float(missed.sum()), missed, pred, *others)

def evaluate(opt, model):
    model.eval()
    totalErr = 0
    count = 0
    for x, y, l, mask in newLoader('val', batch_size=opt.batchsize):
        count += int(l.sum())
        err, _, pred, _, *others = evaluateStep(opt, model, x, y, l, mask)
        totalErr += err
    if opt.drawVars:
        opt.drawVars(x[0], l[0], *tuple(v[0] for v in others))
        print(pred[0])
    return totalErr / count

def initTrain(opt, model, epoch=None):
    opt.optimizer = opt.newOptimizer(opt, model.parameters(), 1e-8)
    if opt.sdt_decay_step > 0:
        opt.scheduler = optim.lr_scheduler.StepLR(opt.optimizer, opt.sdt_decay_step, gamma=0.5)
    else:
        opt.scheduler = optim.lr_scheduler.StepLR(opt.optimizer, 1e6, gamma=1)
    if type(epoch) == int:
        state = torch.load('train.epoch{}.pth'.format(epoch), map_location='cpu')
        opt.optimizer.load_state_dict(state[0])
        opt.scheduler.load_state_dict(state[1])
    else:
        torch.manual_seed(args.rank)
        np.random.seed(args.rank)

def train(opt, model, init=True):
    if init:
        initParameters(opt, model)
        if type(init) == int:
            model.load_state_dict(torch.load('model.epoch{}.pth'.format(epoch), map_location='cpu'))
    model = model.to(opt.device) # need before constructing optimizers
    if init:
        initTrain(opt, model, init)
    if amp:
        model, opt.optimizer = amp.initialize(model, opt.optimizer, opt_level="O2", keep_batchnorm_fp32=False)
    for i in range(opt.scheduler.last_epoch, opt.epochs):
        opt.scheduler.step()
        count = 0
        totalLoss = 0
        model.train()
        opt.optimizer.zero_grad()
        for x, y, l, mask in newLoader('train', batch_size=opt.batchsize, shuffle=True):
            length = int(l.sum())
            count += length
            loss = trainStep(opt, model, x, y, length, mask)
            totalLoss += loss
        valErr = evaluate(opt, model)
        if opt.writer:
            logBoardStep(opt, model)
        print('Epoch #%i | train loss: %.4f | valid error: %.3f | learning rate: %.5f' %
          (opt.scheduler.last_epoch, totalLoss / count, valErr, opt.scheduler.get_lr()[0]))
        if i % 10 == 9:
            saveState(opt, model, opt.scheduler.last_epoch)
    return valErr

def saveState(opt, model, epoch):
    torch.save(model.state_dict(), 'model.epoch{}.pth'.format(epoch))
    torch.save((opt.optimizer.state_dict(), opt.scheduler.state_dict()), 'train.epoch{}.pth'.format(epoch))

def logBoardStep(opt, model):
    step = opt.scheduler.last_epoch
    for name, param in model.named_parameters():
        try:
            opt.writer.add_histogram(name, param.data, step)
        except:
            print(name, param)

torch.manual_seed(args.rank)
np.random.seed(args.rank)
model = Model(opt).to(opt.device)
print('Number of parameters: %i | valid error: %.3f' % (getNelement(model), evaluate(opt, model)))
train(opt, model)
torch.save(model.state_dict(), 'model.epoch{}.pth'.format(opt.scheduler.last_epoch))
