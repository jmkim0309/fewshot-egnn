import sys
import configparser
import torch
import threading
import time
import os


__author__ = 'namju.kim@kakaobrain.com'


_config_time_stamp = 0


class _Opt(object):

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return None

    def __getattr__(self, item):
        return self.__getitem__(item)


def _to_py_obj(x):
    # check boolean first
    if x.lower() in ['true', 'yes', 'on']:
        return True
    if x.lower() in ['false', 'no', 'off']:
        return False
    # from string to python object if possible
    try:
        obj = eval(x)
        if type(obj).__name__ in ['int', 'float', 'tuple', 'list', 'dict', 'NoneType']:
            x = obj
    except:
        pass
    return x


def _parse_config(arg, file):

    # read config file
    config = configparser.ConfigParser()
    config.read(file)
    # traverse sections
    for section in config.sections():
        # traverse items
        opt = _Opt()
        for key in config[section]:
            opt[key] = _to_py_obj(config[section][key])
        # if default section, save items to global scope
        if section.lower() == 'default':
            for k, v in opt.__dict__.items():
                arg[k] = v
        else:
            arg['_'.join(section.split())] = opt


def _parse_config_thread(arg, file):

    global _config_time_stamp

    while True:
        # check timestamp
        stamp = os.stat(file).st_mtime
        if not stamp == _config_time_stamp:
            # update timestamp
            _config_time_stamp = stamp
            # parse config file
            _parse_config(arg, file)
            # print result
            # _print_opts(arg, 'CONFIGURATION CHANGE DETECTED')
        # sleep
        time.sleep(1)


def _print_opts(arg, header):
    print(header, flush=True)
    print('-' * 30, flush=True)
    for k, v in arg.__dict__.items():
        print('%s=%s' % (k, v), flush=True)
    print('-' * 30, flush=True)


def _parse_opts():

    global _config_time_stamp

    # get command line arguments
    arg = _Opt()
    argv = sys.argv[1:]

    # check length
    assert len(argv) % 2 == 0, 'arguments should be paired with the format of --key value'

    # parse args
    for i in range(0, len(argv), 2):

        # check format
        assert argv[i].startswith('--'), 'arguments should be paired with the format of --key value'

        # save argument
        arg[argv[i][2:]] = _to_py_obj(argv[i + 1])

        # check config file
        if argv[i][2:].lower() == 'config':
            _parse_config(arg, argv[i + 1])
            _config_time_stamp = os.stat(argv[i + 1]).st_mtime

    #
    # inject default options
    #

    # device setting
    if arg.device is None:
        arg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arg.device = torch.device(arg.device)
    arg.cuda = arg.device.type == 'cuda'

    # default learning rate
    #arg.lr = 1e-3

    # directories
    arg.log_dir = arg.log_dir or 'asset/log/'
    arg.data_dir = arg.data_dir or 'asset/data/'
    arg.save_dir = arg.save_dir or 'asset/train/'
    arg.log_dir += '' if arg.log_dir.endswith('/') else '/'
    arg.data_dir += '' if arg.data_dir.endswith('/') else '/'
    arg.save_dir += '' if arg.save_dir.endswith('/') else '/'

    # print arg option
    # _print_opts(arg, 'CONFIGURATION')

    # start config file watcher if config is defined
    if arg.config:
        t = threading.Thread(target=_parse_config_thread, args=(arg, arg.config))
        t.daemon = True
        t.start()

    return arg
