import os
import shutil
import time

import numpy as np
from torch.optim import SGD, Adam


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def compute_num_params(model, text=True):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False,
                   lr_scheduler_spec=None, pre_steps=0):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def step(self, text=True):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        self.t_last = t_cur
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
