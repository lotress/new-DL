import os
import argparse
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_known_args()[0]
def opt():pass
if torch.cuda.is_available():
  opt.dtype = torch.half
  opt.device = torch.device('cuda:{}'.format(args.local_rank))
  torch.cuda.set_device(args.local_rank)
  opt.cuda = True
else:
  opt.device = torch.device('cpu')
  opt.dtype = torch.float
  opt.cuda = False
  num_threads = torch.multiprocessing.cpu_count() - 1
  if num_threads > 1:
    torch.set_num_threads(num_threads)
print('Using device ' + str(opt.device))
print('Using default dtype ' + str(opt.dtype))
