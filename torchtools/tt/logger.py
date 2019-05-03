import datetime
import time
from tensorboardX import SummaryWriter
from torchtools import tt


__author__ = 'namju.kim@kakaobrain.com'


# tensorboard writer
_writer = None
_stats_scalar, _stats_image, _stats_audio, _stats_text, _stats_hist = {}, {}, {}, {}, {}

# time stamp
_last_logged = time.time()


# general print wrapper
def log(*args):
    print(*args, flush=True)
    # save to log_file
    if tt.arg.log_file:
        with open(tt.arg.log_dir + tt.arg.log_file, 'a') as f:
            print(*args, flush=True, file=f)


# tensor board writer
def _get_writer():
    global _writer
    if _writer is None:
        # logging directory
        tf_log_dir = tt.arg.log_dir
        tf_log_dir += '' if tf_log_dir.endswith('/') else '/'
        if tt.arg.experiment:
            tf_log_dir += tt.arg.experiment
        tf_log_dir += datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')
        # create writer
        _writer = SummaryWriter(tf_log_dir)
    return _writer


def log_scalar(tag, value, global_step=None):
    _stats_scalar[tag] = (tt.nvar(value), global_step)


def log_audio(tag, audio, global_step=None):
    _stats_audio[tag] = (tt.nvar(audio), global_step)


def log_image(tag, image, global_step=None):
    _stats_image[tag] = (tt.nvar(image), global_step)


def log_text(tag, text, global_step=None):
    _stats_text[tag] = (text, global_step)


def log_hist(tag, values, global_step=None):
    _stats_hist[tag] = (tt.nvar(values), global_step)


def log_step(epoch=None, global_step=None, max_epoch=None, max_step=None):

    global _last_logged, _last_logged_step, _stats_scalar, _stats_image, _stats_audio, _stats_text, _stats_hist

    # logging
    if (tt.arg.log_interval is None and tt.arg.log_step is None) or \
       (tt.arg.log_interval and time.time() - _last_logged >= tt.arg.log_interval) or \
       (tt.arg.log_step and global_step % tt.arg.log_step == 0):

        # update logging time stamp
        _last_logged = time.time()
        _last_logged_step = global_step

        # console output string
        console_out = ''
        if epoch:
            console_out += 'ep: %d' % epoch
            if max_epoch:
                console_out += '/%d' % max_epoch
        if global_step:
            if max_step:
                step = global_step % max_step
                step = max_step if step == 0 else step
                console_out += ' step: %d/%d' % (step, max_step)
            else:
                console_out += ' step: %d' % global_step

        # add stats to tensor board
        for k, v in _stats_scalar.items():
            _get_writer().add_scalar(k, *v)
            # add to console output
            if not k.startswith('weight/') and not k.startswith('gradient/'):
                console_out += ' %s: %f' % (k, v[0])
        for k, v in _stats_image.items():
            _get_writer().add_image(k, *v)
        for k, v in _stats_audio.items():
            _get_writer().add_audio(k, *v)
        for k, v in _stats_text.items():
            _get_writer().add_text(k, *v)
        for k, v in _stats_hist.items():
            _get_writer().add_histogram(k, *v, 'auto')

        # flush
        _get_writer().file_writer.flush()

        # console out
        if len(console_out) > 0:
            log(console_out)

        # clear stats
        _stats_scalar, _stats_image, _stats_audio, _stats_text = {}, {}, {}, {}


def log_weight(model, global_step=None):
    # weight statics
    if tt.arg.log_weight:
        for k, v in model.named_parameters():
            if 'weight' in k:  # only for weight not bias
                log_scalar('weight/' + k, v.norm(), global_step)


def log_gradient(model, global_step=None):
    # gradient statics
    if tt.arg.log_grad:
        for k, v in model.named_parameters():
            if 'weight' in k:  # only for weight not bias
                if v.grad is not None:
                    log_scalar('gradient/' + k, v.grad.norm(), global_step)
