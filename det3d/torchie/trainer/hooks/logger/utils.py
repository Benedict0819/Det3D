import datetime
from collections import OrderedDict
import torch
import torch.distributed as dist
from det3d import torchie

def GetClassNames(trainer):
    if trainer.world_size > 1:
        class_names = trainer.model.module.bbox_head.class_names
    else:
        class_names = trainer.model.bbox_head.class_names
    return class_names

def ConvertToPrecision4(val):
    if isinstance(val, float):
        val = "{:.4f}".format(val)
    elif isinstance(val, list):
        val = [ConvertToPrecision4(v) for v in val]

    return val

def GetMaxMemory(trainer):
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor(
        [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
    )
    if trainer.world_size > 1 and mem_mb.size() > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()

def GetMasterMemory(trainer):
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor(
        [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
    )
    return mem_mb.item()


def GetLogDict(trainer):
    log_dict = OrderedDict()
    # Training mode if the output contains the key time
    mode = "train" if "time" in trainer.log_buffer.output else "val"
    log_dict["mode"] = mode
    log_dict["epoch"] = trainer.epoch + 1
    log_dict["iter"] = trainer.inner_iter + 1
    # Only record lr of the first param group
    log_dict["lr"] = trainer.current_lr()[0]
    log_dict["total_num_points"] = trainer.get_example_stats()["total_num_points"]
    if mode == "train":
        log_dict["time"] = trainer.log_buffer.output["time"]
        log_dict["data_time"] = trainer.log_buffer.output["data_time"]
        # statistic memory
        if torch.cuda.is_available():
            log_dict["memory"] = GetMasterMemory(trainer)
            # log_dict["memory"] = 0
    for name, val in trainer.log_buffer.output.items():
        if name in ["time", "data_time"]:
            continue
        log_dict[name] = val
    return log_dict

def ParseLog(trainer, logger, log_dict):
    tb_log_dicts = OrderedDict()
    log_strs = []

    if trainer.mode == "train":
        log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, total_num_points: {:.0f}, ".format(
            log_dict["epoch"],
            trainer._max_epochs,
            log_dict["iter"],
            len(trainer.data_loader),
            log_dict["lr"],
            log_dict["total_num_points"],
        )
        if "time" in log_dict.keys():
            logger.time_sec_tot += log_dict["time"] * logger.interval
            time_sec_avg = logger.time_sec_tot / (trainer.iter - logger.start_iter + 1)
            eta_sec = time_sec_avg * (trainer.max_iters - trainer.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += f"eta: {eta_str}, "
            log_str += "time: {:.3f}, data_time: {:.3f}, transfer_time: {:.3f}, forward_time: {:.3f}, loss_parse_time: {:.3f} ".format(
                log_dict["time"],
                log_dict["data_time"],
                log_dict["transfer_time"] - log_dict["data_time"],
                log_dict["forward_time"] - log_dict["transfer_time"],
                log_dict["loss_parse_time"] - log_dict["forward_time"],
            )
            log_str += f"memory: {log_dict['memory']}, "
            
    else:
        log_str = "Epoch({}) [{}][{}]\t".format(
            log_dict["mode"], log_dict["epoch"] - 1, log_dict["iter"]
        )
    log_strs.append(log_str)
    tb_log_dicts["learning_rate"] = log_dict["lr"]

    class_names = GetClassNames(trainer)

    for idx, task_class_names in enumerate(class_names):
        log_items = [f"task : {task_class_names}"]
        task_class_names_tb = "__".join(task_class_names)
        log_str = ""
        for name, val in log_dict.items():
            if name in [
                "mode",
                "Epoch",
                "iter",
                "lr",
                "total_num_points",
                "time",
                "data_time",
                "memory",
                "epoch",
                "transfer_time",
                "forward_time",
                "loss_parse_time",
            ]:
                continue

            if isinstance(val, list):
                tb_log_dicts[f"{task_class_names_tb}/{name}/{trainer.mode}"] = val[idx]
                log_items.append(f"{name}: {ConvertToPrecision4(val[idx])}")
            else:
                if isinstance(val, float):
                    tb_log_dicts[f"{task_class_names_tb}/{name}/{trainer.mode}"] = val
                    val = f"{val:.4f}"
                    
                log_items.append(f"{name}: {val}")


        log_str += ", ".join(log_items)
        log_strs.append(log_str)

    if "loss" in log_dict:
        total_loss = sum(log_dict["loss"])
        tb_log_dicts[f"total_loss/{trainer.mode}"] = total_loss
        log_strs.append(f"Total loss: {total_loss:.4f}\n")

    return log_strs, tb_log_dicts

