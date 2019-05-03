import os
import datetime
import time
import pathlib
from torchtools import torch, nn, tt


__author__ = 'namju.kim@kakaobrain.com'


# time stamp
_tic_start = _last_saved = _last_archived = time.time()
# best statics
_best = -100000000.


def tic():
    global _tic_start
    _tic_start = time.time()
    return _tic_start


def toc(tic=None):
    global _tic_start
    if tic is None:
        return time.time() - _tic_start
    else:
        return time.time() - tic


def sleep(seconds):
    time.sleep(seconds)


#
# automatic device-aware torch.tensor
#
def var(data, dtype=None, device=None, requires_grad=False):
    # return torch.tensor(data, dtype=dtype, device=(device or tt.arg.device), requires_grad=requires_grad)
    # the upper code doesn't work, so work around as following. ( maybe bug )
    return torch.tensor(data, dtype=dtype, requires_grad=requires_grad).to((device or tt.arg.device))


def vars(x_list, dtype=None, device=None, requires_grad=False):
    return [var(x, dtype, device, requires_grad) for x in x_list]


# for old torchtools compatibility
def cvar(x):
    return x.detach()


#
# to python or numpy variable(s)
#
def nvar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        x = x.item() if x.dim() == 0 else x.numpy()
    return x


def nvars(x_list):
    return [nvar(x) for x in x_list]


def load_model(model, best=False, postfix=None, experiment=None):
    global _best

    # model file name
    filename = tt.arg.save_dir + '%s.pt' % (experiment or tt.arg.experiment or model.__class__.__name__.lower())
    if postfix is not None:
        filename = filename + '.%s' % postfix

    # load model
    global_step = 0
    if os.path.exists(filename):
        if best:
            global_step, model_state, _best = torch.load(filename + '.best', map_location=lambda storage, loc: storage)
        else:
            global_step, model_state = torch.load(filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state)

    # update best stat
    filename += '.best'
    if os.path.exists(filename):
        _, _, _best = torch.load(filename, map_location=lambda storage, loc: storage)

    return global_step


def save_model(model, global_step, force=False, best=None, postfix=None):
    global _last_saved, _last_archived, _best

    # make directory
    pathlib.Path(tt.arg.save_dir).mkdir(parents=True, exist_ok=True)

    # filename to save
    filename = '%s.pt' % (tt.arg.experiment or model.__class__.__name__.lower())
    if postfix is not None:
        filename = filename + '.%s' % postfix

    # save model
    if force or (tt.arg.save_interval and time.time() - _last_saved >= tt.arg.save_interval) or \
       (tt.arg.save_step and global_step % tt.arg.save_step == 0):
        torch.save((global_step, model.state_dict()), tt.arg.save_dir + filename)
        _last_saved = time.time()

    # archive model
    if (tt.arg.archive_interval and time.time() - _last_archived >= tt.arg.archive_interval) or \
       (tt.arg.archive_step and global_step % tt.arg.archive_step == 0):
        # filename to archive
        if tt.arg.archive_interval:
            filename = filename + datetime.datetime.now().strftime('.%Y%m%d.%H%M%S')
        else:
            filename = filename + '.%d' % global_step
        torch.save((global_step, model.state_dict()), tt.arg.save_dir + filename)
        _last_archived = time.time()

    # save best model
    if best is not None and best > _best:
        _best = best
        filename = filename + '.best'
        torch.save((global_step, model.state_dict(), best), tt.arg.save_dir + filename)


# patch Module
nn.Module.load_model = load_model
nn.Module.save_model = save_model
