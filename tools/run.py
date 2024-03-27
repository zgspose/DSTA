#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.abspath('../'))
from posetimation import get_cfg, update_config
from engine import default_parse_args, DefaultRunner
from mmcv import Config

def setup(args):
    cfg = get_cfg(args)
    cfg1 = Config.fromfile(args.config)
    cfg.update(cfg1)
    update_config(cfg, args)
    cfg.data.test.test_mode = True

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    return cfg


def main():
    args = default_parse_args()
    cfg = setup(args)
    runner = DefaultRunner(cfg, args)
    runner.launch()


if __name__ == '__main__':
    main()
