from common import *
from copy import deepcopy
import json
import numpy as np
from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.logs import TensorboardLogger
from parlai.utils.distributed import is_distributed
from model import Model, predict
from train import opt, initParameters, getParamOptions, nan

def _isFp16(x):
  s = x.lower()
  return 2 if s == 'true' else 0 if s == 'false' else int(x)

class MyAgent(TorchAgent):
  def __init__(self, optAgent, shared=None):
    init_model, is_finetune = self._get_init_model(optAgent, shared)
    super().__init__(optAgent, shared)
    if optAgent.get('numthreads', 1) > 1:
      torch.set_num_threads(1)
    optAgent['gradient_clip'] = opt.maxgrad
    self.criterion = opt.criterion
    self.loss = opt.loss
    self.drawVars = opt.drawVars
    opt.edim = optAgent['embeddingsize']
    opt.vocabsize = len(self.dict)
    opt.__dict__.update(optAgent)
    opt.agent = self
    opt.fp16 = self.fp16
    torch.manual_seed(args.rank)
    np.random.seed(args.rank)
    self.writeVars = 0
    self.vars = {}
    if optAgent['tensorboard_log']:
      self.writeVars, *_ = getWriter(writer=TensorboardLogger(optAgent))
    if self.fp16:
      try:
        from apex import amp
      except ImportError:
        raise ImportError(
          'No fp16 support without apex. Please install it from '
          'https://github.com/NVIDIA/apex'
        )
      self.getParameters = lambda: amp.master_params(self.optimizer)
      self.amp = amp
    else:
      self.getParameters = lambda: self.model.parameters()
    if not shared:
      model = Model(opt)
      self.model = model
      if init_model:
        print('Loading existing model parameters from ' + init_model)
        states = self.load(init_model)
      else:
        states = {}
        initParameters(opt, self.model)
      if self.use_cuda:
        self.model.cuda()
      self.model.train()
      if optAgent.get('numthreads', 1) > 1:
        self.model.share_memory()
      paramOptions = getParamOptions(opt, self.model)
      self.init_optim(paramOptions, states.get('optimizer'), states.get('saved_optim_type', None))
      self.build_lr_scheduler(states, hard_reset=is_finetune)
      if is_distributed():
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.opt['gpu']], broadcast_buffers=False)
      self.reset()
    else:
      self.model = shared['model']
      self.dict = shared['dict']
      if 'optimizer' in shared:
        self.optimizer = shared['optimizer']

  def _get_init_model(self, opt, shared):
    """
    Get model file to initialize with.

    :return:  path to load model from, whether we loaded from `init_model`
              or not
    """
    if shared:
      return None, False
    if opt.get('model_file') and os.path.isfile(opt['model_file'] + '.opt.json'):
      return opt['model_file'], False
    if opt.get('init_model') and os.path.isfile(opt['init_model'] + '.opt.json'):
      return opt['init_model'], True
    return None, False

  def build_dictionary(self):
    """Return the constructed dictionary, which will be set to self.dict."""
    d = super().build_dictionary()
    if 'dict_file' in self.opt:
      d.load(self.opt['dict_file'])
    return d

  def share(self):
    """Share internal states between parent and child instances."""
    shared = super().share()
    if hasattr(self, 'optimizer'):
      shared['optimizer'] = self.optimizer
    return shared

  def reset(self):
    """Reset episode_done."""
    super().reset()
    self.episode_done = True
    return self

  def save(self, path):
    """Save model, options, dict."""
    path = self.opt.get('model_file', None) if path is None else path
    if not path:
      return
    states = self.state_dict()
    if states:
      torch.save(states['model'], path + '.pth')
      del states['model']
      with open(path + '.states', 'wb') as write:
        torch.save(states, write)
    # Parlai expects options to also be saved
    with open(path + '.opt.json', 'w', encoding='utf-8') as handle:
      if hasattr(self, 'model_version'):
        self.opt['model_version'] = self.model_version()
      saved_opts = deepcopy(self.opt)
      if 'interactive_mode' in saved_opts:
        # We do not save the state of interactive mode, it is only decided
        # by scripts or command line.
        del saved_opts['interactive_mode']
      json.dump(self.opt, handle)
      # for convenience of working with jq, make sure there's a newline
      handle.write('\n')

    # force save the dict
    dictPath = self.opt['dict_file'] if 'dict_file' in self.opt else path + '.dict.txt'
    self.dict.save(dictPath, sort=False)

  def load_state_dict(self, state_dict):
    """Load the state dict into model."""
    self.model.load_state_dict(state_dict)
    if self.use_cuda:
      self.model.cuda()

  def load(self, path):
    """Load model, options, dict."""
    statePath = path + '.states'
    states = torch.load(statePath, map_location='cpu') if os.path.isfile(statePath) else {}
    optPath = path + '.opt.json'
    if os.path.isfile(optPath):
      with open(optPath, 'r', encoding='utf-8') as handle:
        self.opt = json.load(handle)
        states['saved_optim_type'] = self.opt['optimizer']
    modelPath = path + '.pth'
    if os.path.isfile(modelPath):
      states['model'] = torch.load(modelPath, map_location='cpu')
      self.load_state_dict(states['model'])
    return states

  def is_valid(self, obs):
    """Override from TorchAgent.
    Check if an observation has no tokens in it."""
    return len(obs.get('text_vec', [])) > 0

  def vectorize(self, *args, **kwargs):
    """
    Make vectors out of observation fields and store in the observation.

    In particular, the 'text' and 'labels'/'eval_labels' fields are
    processed and a new field is added to the observation with the suffix
    '_vec'.
    """
    kwargs['add_start'] = False
    kwargs['add_end'] = False
    return super().vectorize(*args, **kwargs)

  def batchify(self, *args, **kwargs):
    """
    Create a batch of valid observations from an unchecked batch.

    A valid observation is one that passes the lambda provided to the
    function, which defaults to checking if the preprocessed 'text_vec'
    field is present which would have been set by this agent's 'vectorize'
    function.

    Returns a namedtuple Batch. See original definition above for in-depth
    explanation of each field.

    If you want to include additonal fields in the batch, you can subclass
    this function and return your own "Batch" namedtuple: copy the Batch
    namedtuple at the top of this class, and then add whatever additional
    fields that you want to be able to access. You can then call
    super().batchify(...) to set up the original fields and then set up the
    additional fields in your subclass and return that batch instead.

    :param obs_batch:
        List of vectorized observations

    :param sort:
        Default False, orders the observations by length of vectors. Set to
        true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
        vectors if available, otherwise uses the label vectors if available.
    """
    batch = super().batchify(*args, **kwargs)
    if not batch.valid_indices or not len(batch.valid_indices):
      return batch

    batch.done_vec = torch.tensor([(1 if ex.get('episode_done') else 0) for ex in batch.observations], dtype=torch.uint8)
    lengths = batch.text_lengths
    if lengths:
      batch.text_lengths = torch.tensor(lengths)
      text_mask = torch.zeros(batch.text_vec.shape, dtype=torch.uint8)
      for i in range(len(lengths)):
        text_mask[i, :lengths[i]].fill_(1)
      batch.text_mask = text_mask.cuda() if self.use_cuda else text_mask
    return batch

  def init_optim(self, params, optim_states=None, saved_optim_type=None):
    """
    Initialize optimizer with model parameters.
    """
    opt = self.opt

    # set up optimizer args
    lr = opt['learningrate']
    kwargs = {'lr': lr}
    if opt.get('weight_decay'):
      kwargs['weight_decay'] = opt['weight_decay']
    if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
      # turn on momentum for optimizers that use it
      kwargs['momentum'] = opt['momentum']
      if opt['optimizer'] == 'sgd' and opt.get('nesterov', True):
        # for sgd, maybe nesterov
        kwargs['nesterov'] = opt.get('nesterov', True)
      elif opt['optimizer'] == 'qhm':
        # qhm needs a nu
        kwargs['nu'] = opt.get('nus', (0.7,))[0]
    elif opt['optimizer'] == 'adam':
      # turn on amsgrad for adam
      # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
      kwargs['amsgrad'] = True
    elif opt['optimizer'] == 'qhadam':
      # set nus for qhadam
      kwargs['nus'] = opt.get('nus', (0.7, 1.0))
    if opt['optimizer'] in ['adam', 'sparseadam', 'fused_adam', 'adamax', 'qhadam']:
      # set betas for optims that use it
      kwargs['betas'] = opt.get('betas', (0.9, 0.999))
      # set adam optimizer, but only if user specified it
      if opt.get('adam_eps'):
        kwargs['eps'] = opt['adam_eps']

    optim_class = self.optim_opts()[opt['optimizer']]
    self.optimizer = optim_class(params, **kwargs)
    if self.fp16:
      self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level="O{}".format(int(self.fp16)))

    if optim_states and saved_optim_type != opt['optimizer']:
      print('WARNING: not loading optim state since optim class changed.')
    elif optim_states:
      optimstate_fp16 = 'loss_scaler' in optim_states
      if self.fp16 and optimstate_fp16:
        optim_states['loss_scaler'] = self.optimizer.state_dict()['loss_scaler']
      elif optimstate_fp16 and not self.fp16:
        optim_states = optim_states['optimizer_state_dict']
      elif not optimstate_fp16 and self.fp16:
        self.optimizer.optimizer.load_state_dict(optim_states)
        return

      # finally, try to actually load the optimizer state
      try:
        self.optimizer.load_state_dict(optim_states)
      except ValueError:
        print('WARNING: not loading optim state since model params changed.')

  def build_lr_scheduler(self, states=None, hard_reset=False):
    """
    Create the learning rate scheduler, and assign it to self.scheduler.
    """
    # first make sure there are no null pointers
    if states is None:
      states = {}
    optimizer = self.optimizer

    warmup_updates = self.opt.get('warmup_updates', -1)
    updates_so_far = states.get('number_training_updates', 0)
    if warmup_updates > 0 and (updates_so_far < warmup_updates or hard_reset):

      def _warmup_lr(step):
        start = self.opt['warmup_rate']
        end = 1.0
        progress = min(1.0, step / self.opt['warmup_updates'])
        lr_mult = start + (end - start) * progress
        return lr_mult

      self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr)
    else:
      self.warmup_scheduler = None

    patience = self.opt.get('lr_scheduler_patience', 3)
    decay = self.opt.get('lr_scheduler_decay', 0.5)

    if self.opt.get('lr_scheduler') == 'none':
      self.scheduler = None
    elif decay == 1.0:
      warn_once(
        "Your LR decay is set to 1.0. Assuming you meant you wanted "
        "to disable learning rate scheduling. Adjust --lr-scheduler-decay "
        "if this is not correct."
      )
      self.scheduler = None
    elif self.opt.get('lr_scheduler') == 'reduceonplateau':
      self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=decay, patience=patience, verbose=True
      )
    elif self.opt.get('lr_scheduler') == 'fixed':
      self.scheduler = optim.lr_scheduler.StepLR(optimizer, patience, gamma=decay)
    elif self.opt.get('lr_scheduler') == 'invsqrt':
      if self.opt.get('warmup_updates', -1) <= 0:
        raise ValueError('--lr-scheduler invsqrt requires setting --warmup-updates')
      warmup_updates = self.opt['warmup_updates']
      decay_factor = np.sqrt(max(1, warmup_updates))

      def _invsqrt_lr(step):
        return decay_factor / np.sqrt(max(1, step))

      self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, _invsqrt_lr)
    else:
      raise ValueError(
        "Don't know what to do with lr_scheduler '{}'".format(self.opt.get('lr_scheduler'))
      )

    # time to load LR state from the checkpoint, if possible.
    if (
      # there is already an old LR scheduler saved on disk
      states
      and
      # and the old LR scheduler is different
      states.get('lr_scheduler_type') != self.opt['lr_scheduler']
      and
      # and we're not already using a fresh scheduler
      not hard_reset
    ):
      # the LR scheduler changed, start things fresh
      warn_once("LR scheduler is different from saved. Starting fresh!")
      hard_reset = True

    if hard_reset:
      # We're not going to use the LR schedule, let's just exit
      return

    # do the actual loading (if possible)
    if 'number_training_updates' in states:
      self._number_training_updates = states['number_training_updates']
    if self.scheduler and 'lr_scheduler' in states:
      self.scheduler.load_state_dict(states['lr_scheduler'])
    if states.get('warmup_scheduler') and getattr(self, 'warmup_scheduler', None):
      self.warmup_scheduler.load_state_dict(states['warmup_scheduler'])

  def backward(self, loss):
    """
    Perform a backward pass.
    """
    update_freq = self.opt.get('update_freq', 1)
    if update_freq > 1:
      # gradient accumulation, but still need to average across the minibatches
      loss = loss / update_freq
      # we're doing gradient accumulation, so we don't only want to step
      # every N updates instead
      self._number_grad_accum = (self._number_grad_accum + 1) % update_freq

    if self.fp16:
      delay_unscale = update_freq > 1 and self._number_grad_accum > 0
      with self.amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
        scaled_loss.backward()
    else:
      loss.backward()

  def update_params(self):
    """
    Perform step of optimization.
    """
    update_freq = self.opt.get('update_freq', 1)
    if update_freq > 1 and self._number_grad_accum > 0:
      return

    if hasattr(opt, 'gradF'):
      opt.gradF(self.model, self.getParameters())

    if self.opt.get('gradient_clip', -1) > 0:
      grad_norm = torch.nn.utils.clip_grad_norm_(self.getParameters(), self.opt['gradient_clip'])
      self.metrics['gnorm'] += grad_norm
      self.metrics['clip'] += float(grad_norm > self.opt['gradient_clip'])

    self.metrics['updates'] += 1
    self.optimizer.step()

    # keep track up number of steps, compute warmup factor
    self._number_training_updates += 1

    # compute warmup adjustment if needed
    if self.opt.get('warmup_updates', -1) > 0:
      if not hasattr(self, 'warmup_scheduler'):
        raise RuntimeError('Looks like you forgot to call build_lr_scheduler')
      if self._is_lr_warming_up():
        self.warmup_scheduler.step(epoch=self._number_training_updates)

    if self.opt.get('lr_scheduler') == 'invsqrt' and not self._is_lr_warming_up():
      # training step scheduler
      self.scheduler.step(self._number_training_updates)

    if hasattr(opt, 'paraF'):
      opt.paraF(opt, self.model)

  def train_step(self, batch):
    """Process batch of inputs and targets and train on them.

    :param batch: parlai.core.torch_agent.Batch, contains tensorized
                  version of observations.
    """
    if batch.text_vec is None:
      return
    self.is_training = True
    self.model.train()
    self.zero_grad()
    output = self.model(batch.text_vec, batch.text_mask)
    loss = self.loss(self, self.model, batch.label_vec, *output).sum()
    if torch.allclose(loss, nan, equal_nan=True):
      raise Exception('Loss returns NaN')
    self.backward(loss)
    self.update_params()
    count = int(batch.text_lengths.sum())
    self.metrics['count'] += count
    self.metrics['loss.sum'] += float(loss)
    return # omit response for speed
    #pred = predict(output[0], batch.text_lengths, batch.text_vec, batch.text_mask)
    #return Output(text=pred)

  def eval_step(self, batch):
    """Process batch of inputs.

    If the batch includes labels, calculate validation metrics as well.

    :param batch: parlai.core.torch_agent.Batch, contains tensorized
                  version of observations.
    """
    if batch.text_vec is None:
      return
    self.is_training = False
    self.model.eval()
    output = self.model(batch.text_vec, batch.text_mask)
    if batch.label_vec is not None:
      # Interactive mode won't have a gold label
      missed = self.criterion(batch.label_vec, output[0], batch.text_mask)
      self.metrics['error.sum'] += float(missed.sum())
      self.metrics['eval_exs'] += int(batch.text_lengths.sum())

    pred = predict(output[0], batch.text_lengths, batch.text_vec, batch.text_mask)
    text = self._v2t(batch.text_vec[0])
    self.vars = (text, pred[0], batch.text_vec[0], int(batch.text_lengths[0]), *tuple(v[0] for v in output[2:]))
    return Output(text=pred)

  def report(self):
    """Return metrics calculated by the model."""
    metrics = super().report()
    if 'loss.sum' in self.metrics:
      count = self.metrics['count'] if 'count' in self.metrics and self.metrics['count'] else 1
      self.metrics['loss'] = self.metrics['loss.sum'] / count
    metrics['loss'] = self.metrics['loss']
    metrics['error'] = self.metrics['error.sum'] / (self.metrics['eval_exs'] if self.metrics['eval_exs'] else 1)
    metrics['accuracy'] = 1. - metrics['error']
    if self.writeVars:
      self.writeVars({'loss': metrics['loss']},
        histograms=dict(self.model.named_parameters()),
        n=self.scheduler.last_epoch)
    if len(self.vars):
      if self.drawVars:
        self.drawVars(*self.vars[2:])
      if len(self.vars) > 1 and self.writeVars:
        self.writeVars(images={'hidden': self.vars[4].unqueeze(1)},
          n=self.scheduler.last_epoch)
      if type(self.vars[0]) == str:
        print(self.vars[0], self.vars[1])
      self.vars = []
    return metrics

  def reset_metrics(self):
    """Reset metrics calculated by the model back to zero."""
    super().reset_metrics()
    self.metrics['loss'] = 0.
    self.metrics['loss.sum'] = 0.
    self.metrics['error.sum'] = 0.
    self.metrics['count'] = 0
    self.metrics['eval_exs'] = 0

  def receive_metrics(self, metrics_dict):
    """Update lr scheduler with validation loss."""
    return super().receive_metrics(metrics_dict)

  @classmethod
  def add_cmdline_args(cls, argparser):
    """Add command-line arguments specifically for this agent."""
    super(MyAgent, cls).add_cmdline_args(argparser)

    agent = argparser.add_argument_group('Arguments')
    agent.add_argument('-esz', '--embeddingsize', type=int, default=16,
                        help='size of the token embeddings')
    agent.add_argument('-dr', '--dropout', type=float, default=0.0,
                        help='dropout rate')
    agent.add_argument('--fp16', type=_isFp16, default=2,
                        help='Amp fp16 optimization level')
    argparser.set_defaults(split_lines=True)
    MyAgent.dictionary_class().add_cmdline_args(argparser)
    return agent
