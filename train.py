from common import *
import torch.optim as optim
from data import newLoader
from model import Model, predict
from option import option
if amp:
    from apex.optimizers import FusedAdam
getNelement = lambda model: sum(map(lambda p: p.nelement(), model.parameters()))
l1Reg = lambda acc, cur: acc + cur.abs().sum(dtype=torch.float)
l2Reg = lambda acc, cur: acc + (cur * cur).sum(dtype=torch.float)
nan = torch.tensor(float('nan'), device=opt.device)
toDevice = lambda a, device: tuple(map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, a))
detach0 = lambda x: x[0].detach() if isinstance(x, torch.Tensor) else x[0]

opt.batchsize = 1
opt.epochs = 1
opt.maxgrad = 1. # max gradient
opt.dropout = 0
opt.learningrate = 0.001 # initial learning rate
opt.sdt_decay_step = 10 # how often to reduce learning rate
opt.criterion = lambda y, out, mask, *args: F.mse_loss(out, y) # criterion for evaluation
opt.loss = lambda opt, model, y, out, *args: F.mse_loss(out, y) # criterion for loss function
opt.newOptimizer = (lambda opt, params, _: FusedAdam(params, lr=opt.learningrate)) if amp else lambda opt, params, eps: optim.Adam(params, lr=opt.learningrate, amsgrad=True, eps=eps)
opt.writer = 0 # TensorBoard writer
opt.drawVars = 0
opt.reset_parameters = 0
opt.toImages = 0
opt.__dict__.update(option)

def initParameters(opt, model):
    for m in model.modules():
        if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.PReLU):
            nn.init.constant_(next(m.parameters()), 1)
        if opt.reset_parameters:
            opt.reset_parameters(opt, model)
#    if hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding):
#        model.embedding.weight.data[2:] = torch.load(word2vecPath)

def getParamOptions(opt, model, *config):
    res = []
    s = set()
    base = opt.learningrate
    for m, k in config:
        s = s.union(set(m.parameters()))
        res.append(dict(params=m.parameters(), lr=k * base))
    res.append({'params': filter(lambda p: not p in s, model.parameters())})
    return res

getParameters = (lambda opt, _: amp.master_params(opt.optimizer)) if amp else lambda _, model: model.parameters()
backward = lambda loss, _: loss.backward()
if torch.cuda.is_available():
    def backward(loss, opt):
        with amp.scale_loss(loss, opt.optimizer) as scaled_loss:
            scaled_loss.backward()

def trainStep(opt, model, x, y, length, *args):
    opt.optimizer.zero_grad()
    x = x.to(opt.device, non_blocking=True)
    label = y.to(opt.device, non_blocking=True)
    args = toDevice(args, opt.device)
    loss = opt.loss(opt, model, label, *model(x, *args)).sum()
    if torch.allclose(loss, nan, equal_nan=True):
        raise Exception('Loss returns NaN')
    backward(loss, opt)
    if hasattr(opt, 'gradF'):
        opt.gradF(model)
    nn.utils.clip_grad_value_(getParameters(opt, model), opt.maxgrad)
    opt.optimizer.step()
    return float(loss)

def evaluateStep(opt, model, x, y, l, *args):
    args = toDevice(args, opt.device)
    out, *others = model(x.to(opt.device, non_blocking=True), *args)
    pred = predict(out, l)
    if isinstance(pred, torch.Tensor):
        y = y.to(pred)
    missed = opt.criterion(y, out, *args)
    return (float(missed.sum()), missed, pred, *others)

def evaluate(opt, model):
    model.eval()
    totalErr = 0
    count = 0
    for x, y, l, *args in newLoader('val', batch_size=opt.batchsize):
        count += int(l.sum())
        err, _, pred, _, *others = evaluateStep(opt, model, x, y, l, *args)
        totalErr += err
    vs = tuple(map(detach0, others))
    if opt.drawVars:
        opt.drawVars(x[0], l[0], *vs)
        print(pred[0])
    return totalErr / count, opt.toImages(*vs) if opt.toImages else {}

def initTrain(opt, model, epoch=None):
    paramOptions = getParamOptions(opt, model)
    eps = 1e-4 if opt.fp16 else 1e-8
    opt.optimizer = opt.newOptimizer(opt, paramOptions, eps)
    if opt.sdt_decay_step > 0:
        gamma = opt.gamma if hasattr(opt, 'gamma') else .5
        opt.scheduler = optim.lr_scheduler.StepLR(opt.optimizer, opt.sdt_decay_step, gamma=gamma)
    else:
        opt.scheduler = None
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
        model, opt.optimizer = amp.initialize(model, opt.optimizer, opt_level="O{}".format(opt.fp16))
    for i in range(opt.scheduler.last_epoch, opt.epochs):
        count = 0
        totalLoss = 0
        model.train()
        for x, y, l, *args in newLoader('train', batch_size=opt.batchsize, shuffle=True):
            length = int(l.sum())
            count += length
            loss = trainStep(opt, model, x, y, length, *args)
            totalLoss += loss
        if opt.scheduler:
            opt.scheduler.step()
        valErr, vs = evaluate(opt, model)
        avgLoss = totalLoss / count
        if opt.writer:
            opt.writer({'loss': avgLoss}, images=vs, histograms=dict(model.named_parameters()), n=opt.scheduler.last_epoch)
        print('Epoch #%i | train loss: %.4f | valid error: %.3f | learning rate: %.5f' %
          (opt.scheduler.last_epoch, avgLoss, valErr, opt.scheduler.get_lr()[0]))
        if i % 10 == 9:
            saveState(opt, model, opt.scheduler.last_epoch)
    return valErr

def saveState(opt, model, epoch):
    torch.save(model.state_dict(), 'model.epoch{}.pth'.format(epoch))
    torch.save((opt.optimizer.state_dict(), opt.scheduler.state_dict()), 'train.epoch{}.pth'.format(epoch))

if __name__ == '__main__':
    torch.manual_seed(args.rank)
    np.random.seed(args.rank)
    model = Model(opt).to(opt.device)
    print('Number of parameters: %i | valid error: %.3f' % (getNelement(model), evaluate(opt, model)[0]))
    train(opt, model)
    torch.save(model.state_dict(), 'model.epoch{}.pth'.format(opt.scheduler.last_epoch))
