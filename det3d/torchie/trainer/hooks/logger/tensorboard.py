import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook
from .utils import ParseLog, GetLogDict
import numpy as np

class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.start_iter = trainer.iter
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        # for var in trainer.log_buffer.output:
        #     if var in ["time", "data_time"]:
        #         continue
        #     tag = "{}/{}".format(var, trainer.mode)
        #     record = trainer.log_buffer.output[var]
        #     if isinstance(record, str):
        #         self.writer.add_text(tag, record, trainer.iter)
        #     else:
        #         print(trainer.log_buffer.output[var])
        #         self.writer.add_scalar(
        #             tag, np.asarray(trainer.log_buffer.output[var]), trainer.iter
        #         )

        log_dict = GetLogDict(trainer)
        _, tb_log_dicts = ParseLog(trainer, self, log_dict)
        for k, v in tb_log_dicts.items():
            if isinstance(v, str):
                self.writer.add_text(k, v, trainer.iter)
            elif isinstance(v, list):
                for idx, _v in enumerate(v):
                    self.writer.add_scalar(
                        k+f"/{idx}", np.asarray(_v), trainer.iter
                    )
            else:
                self.writer.add_scalar(
                    k, np.asarray(v), trainer.iter
                )


    @master_only
    def after_run(self, trainer):
        self.writer.close()

