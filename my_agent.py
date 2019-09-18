from common import *
from copy import deepcopy
import json
import numpy as np
from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.logs import TensorboardLogger
from parlai.core.distributed_utils import is_distributed
from model import Model, predict
from train import opt, initParameters, logBoardStep, nan

def _fixDict(d):
    key0 = next(iter(d.keys()))
    d.freq[''] = int(key0)
    d.ind2tok[0] = ''
    d.tok2ind[''] = 0
    del d.freq[key0]
    del d.tok2ind[key0]
    return d

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
        torch.manual_seed(args.rank)
        np.random.seed(args.rank)
        self.writer = TensorboardLogger(optAgent) if optAgent['tensorboard_log'] else 0
        if not shared:
            model = Model(opt)
            self.model = model
            if init_model:
                print('Loading existing model parameters from ' + init_model)
                states = self.load(init_model)
            else:
                states = {}
                initParameters(opt, self.model)
            self.model = self.model.to(opt.device)
            if self.fp16:
                self.model = self.model.half()
            self.model.train()
            if optAgent.get('numthreads', 1) > 1:
                self.model.share_memory()
            self.init_optim(model.parameters(), states.get('optimizer'), states.get('saved_optim_type', None))
            self.build_lr_scheduler(states, hard_reset=is_finetune)
            if is_distributed():
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.opt['gpu']], broadcast_buffers=False)
        else:
            self.model = shared['model']
            self.dict = shared['dict']
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        self.reset()

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
        self.opt.update(dict(dict_nulltoken='', dict_starttoken='', dict_endtoken='', dict_unktoken=''))
        d = super().build_dictionary()
        if 'dict-file' in self.opt:
            d.load(self.opt['dict-file'])
        return d # _fixDict(d)

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
        dictPath = self.opt['dict-file'] if 'dict-file' in self.opt else path + '.dict.txt'
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
        self.metrics['count'] += int(batch.text_lengths.sum())
        self.metrics['loss.sum'] += float(loss)
        return # omit response for speed
        #pred = predict(output[0], batch.text_lengths)
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

        pred = predict(output[0], batch.text_lengths)
        text = self._v2t(batch.text_vec[0])
        self.metrics['pred'] = (text, pred[0])
        self.metrics['vars'] = (batch.text_vec[0], int(batch.text_lengths[0]), *tuple(v[0] for v in output[2:]))
        return Output(text=pred)

    def report(self):
        """Return metrics calculated by the model."""
        metrics = super().report()
        metrics['loss'] = self.metrics['loss']
        metrics['error'] = self.metrics['error.sum'] / (self.metrics['eval_exs'] if self.metrics['eval_exs'] else 1)
        metrics['accuracy'] = 1. - metrics['error']
        if self.writer:
            logBoardStep(self, self.model)
        if self.drawVars and 'vars' in self.metrics:
            self.drawVars(*self.metrics['vars'])
        if 'pred' in self.metrics:
            print(self.metrics['pred'])
        return metrics

    def reset_metrics(self):
        """Reset metrics calculated by the model back to zero."""
        if 'loss.sum' in self.metrics:
            self.metrics['loss'] = self.metrics['loss.sum'] / (self.metrics['count'] if self.metrics['count'] else 1)
        super().reset_metrics()
        self.metrics['loss.sum'] = 0.
        self.metrics['error.sum'] = 0.
        if 'pred' in self.metrics:
            del self.metrics['pred']
        if 'vars' in self.metrics:
            del self.metrics['vars']
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
        argparser.set_defaults(split_lines=True, fp16=True)
        MyAgent.dictionary_class().add_cmdline_args(argparser)
        return agent
