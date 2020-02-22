import datetime
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist
from det3d import torchie

from .base import LoggerHook
from .utils import ParseLog, GetLogDict

class TextLoggerHook(LoggerHook):
    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)

    def before_run(self, trainer):
        super(TextLoggerHook, self).before_run(trainer)
        self.start_iter = trainer.iter
        self.json_log_path = osp.join(
            trainer.work_dir, "{}.log.json".format(trainer.timestamp)
        )

    def _log_info(self, log_dict, trainer):

        log_strs, _ = ParseLog(trainer, self, log_dict)
        for log_str in log_strs:
            trainer.logger.info(log_str)
        

    def _dump_log(self, log_dict, trainer):
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)

        if trainer.rank == 0:
            with open(self.json_log_path, "a+") as f:
                torchie.dump(json_log, f, file_format="json")
                f.write("\n")

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, trainer):
        log_dict = GetLogDict(trainer)

        self._log_info(log_dict, trainer)
        self._dump_log(log_dict, trainer)
