# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import time
import argparse
import logging
import tqdm
import os
import sys
import numpy as np
import weakref
from collections import OrderedDict
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import torch
from torch.nn.parallel import DistributedDataParallel

from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.datasets import build_xmodaler_train_loader, build_xmodaler_valtest_loader
from xmodaler.modeling import build_model
from xmodaler.optim import build_optimizer
from xmodaler.lr_scheduler import build_lr_scheduler
from xmodaler.evaluation import build_evaluation
from xmodaler.losses import build_losses
from xmodaler.config import kfg
from xmodaler.utils import comm
from xmodaler.utils.collect_env import collect_env_info
from xmodaler.utils.env import TORCH_VERSION, seed_all_rng
from xmodaler.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, get_event_storage
from xmodaler.utils.file_io import PathManager
from xmodaler.utils.logger import setup_logger
from xmodaler.engine.ema import ModelEma

from . import hooks
from .train_loop import TrainerBase
from .build import ENGINE_REGISTRY
import matplotlib.pyplot as plt

__all__ = [
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultTrainer",
]

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by X-modaler users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser    

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the X-modaler logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

@ENGINE_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`DefaultTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`DefaultTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`DefaultTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in X-modaler.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (XmodalerCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("xmodaler")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        
        # Initialize model, optimizer, and data loaders
        model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, model)
        self.train_data_loader = self.build_train_loader(cfg)
        self.val_data_loader = self.build_val_loader(cfg)
        self.test_data_loader = self.build_test_loader(cfg)
    
        # Initialize iteration tracking variables
        if isinstance(self.train_data_loader, list):
            self.iters_per_epoch_list = [len(loader) for loader in self.train_data_loader]
            self._train_data_loader_iter_list = [iter(loader) for loader in self.train_data_loader]
    
            self.iters_per_epoch = len(self.train_data_loader[0])
            self._train_data_loader_iter = iter(self.train_data_loader[0])
        else:
            self.iters_per_epoch = len(self.train_data_loader)
            self._train_data_loader_iter = iter(self.train_data_loader)
    
        # Initialize evaluator for validation and test data (optional)
        if self.val_data_loader is not None:
            self.val_evaluator = build_evaluation(cfg, cfg.INFERENCE.VAL_ANNFILE, None)
        else:
            self.val_evaluator = None
    
        if self.test_data_loader is not None:
            self.test_evaluator = build_evaluation(cfg, cfg.INFERENCE.TEST_ANNFILE, cfg.OUTPUT_DIR)
        else:
            self.test_evaluator = None
    
        # Loss function and scheduler setup
        self.losses = self.build_losses(cfg)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, self.iters_per_epoch)
        self.ss_prob = 0.0
        
        # For training, wrap with DDP (DistributedDataParallel)
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, find_unused_parameters=True, 
                device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        
        self.model = model
        self.model.train()
    
        if cfg.MODEL.USE_EMA:
            self.ema = ModelEma(model, cfg.MODEL.EMA_DECAY)
        else:
            self.ema = None
    
        self.checkpointer = XmodalerCheckpointer(
            # Checkpointing setup
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
    
        # Configuration and iteration management
        self.cfg = cfg
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.EPOCH * self.iters_per_epoch
        self.epoch = 0                     # 当前是第几个 epoch
        self._epoch_loss_sum = 0.0         # 当前 epoch 的 loss 累加
        self._epoch_loss_cnt = 0           # 当前 epoch 内的 step 数
        self.loss_history = []             # 每个 epoch 的平均 loss
        self.bleu4_history = []            # 每个 epoch 的 BLEU-4
        # Initialize lists to keep track of loss and BLEU-4 values for each epoch
        self.loss_history = []  # Stores the average loss per epoch
        self.bleu4_history = []  # Stores the BLEU-4 scores per epoch
        
        self.register_hooks(self.build_hooks())


    def resume_or_load(self, resume=True):
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if self.cfg.MODEL.USE_EMA:
            self.ema = ModelEma(self.model, self.cfg.MODEL.EMA_DECAY)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.ScheduledSampling(
                start_iter = cfg.SCHEDULED_SAMPLING.START_EPOCH * self.iters_per_epoch, 
                inc_every_iter = cfg.SCHEDULED_SAMPLING.INC_EVERY_EPOCH * self.iters_per_epoch, 
                inc_prob = cfg.SCHEDULED_SAMPLING.INC_PROB, 
                max_prob = cfg.SCHEDULED_SAMPLING.MAX_PROB
            ),
            hooks.ModelWeightsManipulating()
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD * self.iters_per_epoch))

        def test_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.test_data_loader, self.test_evaluator, epoch)
            return eval_results

        def test_ema_and_save_results(epoch):
            logger = logging.getLogger(__name__)
            logger.info("Test EMA")

            eval_results = self.test(self.cfg, self.ema.module, self.test_data_loader, self.test_evaluator, epoch)
            return eval_results

        def val_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.val_data_loader, self.val_evaluator, epoch)
            return eval_results

        def val_ema_and_save_results(epoch):
            logger = logging.getLogger(__name__)
            logger.info("VAL EMA")

            eval_results = self.test(self.cfg, self.ema.module, self.val_data_loader, self.val_evaluator, epoch)
            return eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        if self.val_data_loader is not None:
            ret.append(
                hooks.EvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD, 
                    eval_start = cfg.INFERENCE.VAL_EVAL_START,
                    eval_function = val_and_save_results, 
                    iters_per_epoch = self.iters_per_epoch,
                    stage = 'val',
                    multi_gpu_eval=(cfg.ENGINE.NAME.startswith("SingleStreamRetrieval"))
                ))
            if self.ema is not None:
                ret.append(
                    hooks.EvalHook(
                        eval_period = cfg.SOLVER.EVAL_PERIOD, 
                        eval_start = cfg.INFERENCE.VAL_EVAL_START,
                        eval_function = val_ema_and_save_results, 
                        iters_per_epoch = self.iters_per_epoch,
                        stage = 'val',
                        multi_gpu_eval=(cfg.ENGINE.NAME.startswith("SingleStreamRetrieval"))
                    ))

        if self.test_data_loader is not None:
            ret.append(
                hooks.EvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD, 
                    eval_start = cfg.INFERENCE.TEST_EVAL_START,
                    eval_function = test_and_save_results, 
                    iters_per_epoch = self.iters_per_epoch,
                    stage = 'test',
                    multi_gpu_eval=(cfg.ENGINE.NAME.startswith("SingleStreamRetrieval"))
                ))
            if self.ema is not None:
                ret.append(
                    hooks.EvalHook(
                        eval_period = cfg.SOLVER.EVAL_PERIOD, 
                        eval_start = cfg.INFERENCE.TEST_EVAL_START,
                        eval_function = test_ema_and_save_results, 
                        iters_per_epoch = self.iters_per_epoch,
                        stage = 'test',
                        multi_gpu_eval=(cfg.ENGINE.NAME.startswith("SingleStreamRetrieval"))
                    ))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.WRITE_PERIOD))
        return ret

    def build_writers(self):
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_xmodaler_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        return build_xmodaler_valtest_loader(cfg, stage='test')

    @classmethod
    def build_val_loader(cls, cfg):
        return build_xmodaler_valtest_loader(cfg, stage='val')

    @classmethod
    def build_losses(cls, cfg):
        return build_losses(cfg)

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        pass

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            ret["ema"] = self.ema.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({ k: v.detach().cpu().item() })
            else:
                metrics_dict.update({ k: v })
        #metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        results = []
        with torch.no_grad():
            for data in tqdm.tqdm(test_data_loader):
                data = comm.unwrap_model(model).preprocess_batch(data)
                ids = data[kfg.IDS]

                num_knowledge = data['ATT_FEATS'].shape[0]
                foreground_knowledge_dict = np.zeros((num_knowledge, 5, 768), dtype=np.float32)
                foreground_knowledge = np.load('./mscoco_dataset/foreground_val.npy', allow_pickle=True).item()
                ids_list = data['IDS']
                for i, id_str in enumerate(ids_list):
                    id_int = int(id_str)
                    if id_int in foreground_knowledge:
                        value = foreground_knowledge[id_int]
                        foreground_knowledge_dict[i] = value
                data['FOREGROUND_KNOWLEDGE'] = torch.tensor(foreground_knowledge_dict, device=torch.device('cuda'))
                background_knowledge_dict = np.zeros((num_knowledge, 3, 768), dtype=np.float32)
                background_knowledge = np.load('./mscoco_dataset/background_val.npy', allow_pickle=True).item()
                for i, id_str in enumerate(ids_list):
                    id_int = int(id_str)
                    if id_int in background_knowledge:
                        value = background_knowledge[id_int]
                        background_knowledge_dict[i] = value
                data['BACKGROUND_KNOWLEDGE'] = torch.tensor(background_knowledge_dict, device=torch.device('cuda'))
                KNOWLEDGE_MASKS = torch.zeros(num_knowledge,1,1,8).to(torch.device('cuda'))
                data['KNOWLEDGE_MASKS'] = KNOWLEDGE_MASKS
                data['KNOWLEDGE'] = torch.cat((data['FOREGROUND_KNOWLEDGE'], data['BACKGROUND_KNOWLEDGE']), dim=1)

                if cfg.INFERENCE.GENERATION_MODE == True:
                    res = model(data, use_beam_search=True, output_sents=True)
                else:
                    res = model(data)

                outputs = res[kfg.OUTPUT]
                for id, output in zip(ids, outputs):
                    results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output})

        if evaluator is not None:
            eval_res = evaluator.eval(results, epoch)
        else:
            eval_res = ''
        model.train()
        return eval_res

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
    
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter // self.iters_per_epoch)
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
    
        data_time = time.perf_counter() - start
    
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        data[kfg.SS_PROB] = self.ss_prob
    
        # Apply the knowledge part of the model (foreground and background)
        knowledge_num = data['ATT_FEATS'].shape[0]
        foreground_knowledge_dict = np.zeros((knowledge_num, 5, 768), dtype=np.float32)
        foreground_knowledge = np.load('./mscoco_dataset/foreground_train.npy', allow_pickle=True).item()
        ids_list = data['IDS']
        for i, id_str in enumerate(ids_list):
            id_int = int(id_str)
            if id_int in foreground_knowledge:
                value = foreground_knowledge[id_int]
                foreground_knowledge_dict[i] = value
        data['FOREGROUND_KNOWLEDGE'] = torch.tensor(foreground_knowledge_dict, device=torch.device('cuda'))
    
        background_knowledge_dict = np.zeros((knowledge_num, 3, 768), dtype=np.float32)
        background_knowledge = np.load('./mscoco_dataset/background_train.npy', allow_pickle=True).item()
        for i, id_str in enumerate(ids_list):
            id_int = int(id_str)
            if id_int in background_knowledge:
                value = background_knowledge[id_int]
                background_knowledge_dict[i] = value
        data['BACKGROUND_KNOWLEDGE'] = torch.tensor(background_knowledge_dict, device=torch.device('cuda'))
        KNOWLEDGE_MASKS = torch.zeros(knowledge_num, 1, 1, 8).to(torch.device('cuda'))
        data['KNOWLEDGE_MASKS'] = KNOWLEDGE_MASKS
        data['KNOWLEDGE'] = torch.cat((data['FOREGROUND_KNOWLEDGE'], data['BACKGROUND_KNOWLEDGE']), dim=1)
    
        # Forward pass and loss calculation
        outputs_dict = self.model(data)
    
        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
    
        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)
    
        self.optimizer.zero_grad()
        losses.backward()
    
        self._write_metrics(losses_dict, data_time)
    
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)
    
        # Track loss for this epoch
        self._epoch_loss_sum += float(losses.item())
        self._epoch_loss_cnt += 1
    
        # At the end of each epoch: record the loss and evaluate BLEU-4
        if (self.iter + 1) % self.iters_per_epoch == 0:
            self.epoch += 1
            avg_loss = self._epoch_loss_sum / max(1, self._epoch_loss_cnt)
            self.loss_history.append(avg_loss)  # Store loss for plotting
            self._epoch_loss_sum = 0.0
            self._epoch_loss_cnt = 0
    
            # Evaluate the model on validation/test data to get BLEU-4
            eval_res = self.test(self.cfg, self.model, self.test_data_loader, self.test_evaluator, self.epoch)
            bleu4 = 0.0
            if isinstance(eval_res, dict):
                bleu4 = eval_res.get('Bleu_4', 0.0)
            if bleu4 > 1.0:  # Normalize if necessary (for BLEU-4)
                bleu4 = bleu4 / 100.0
            self.bleu4_history.append(bleu4)  # Store BLEU-4 score for plotting
    
            # Update the plot with loss and BLEU-4 curve
            self._plot_curves()
    def _plot_curves(self, save_path='loss_bleu4_curve.png'):
        if len(self.loss_history) == 0:
            return
    
        import matplotlib.pyplot as plt
    
        epochs = list(range(1, len(self.loss_history) + 1))
    
        fig, ax1 = plt.subplots(figsize=(8, 3))
    
        # ===== 左 y 轴：Loss（红色），范围 0~10 =====
        ax1.plot(epochs, self.loss_history, label='Loss', linewidth=1.5, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='red')
        ax1.set_xlim(1, epochs[-1])
        ax1.set_ylim(0.0, 10.0)          # 固定 0~10
        ax1.tick_params(axis='y', labelcolor='red')
    
        # ===== 右 y 轴：BLEU-4（蓝色），范围 0~1 =====
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.bleu4_history, label='BLEU-4', linewidth=1.5, color='blue')
        ax2.set_ylabel('BLEU-4', color='blue')
        ax2.set_ylim(0.0, 1.0)           # 固定 0~1
        ax2.tick_params(axis='y', labelcolor='blue')
    
        # 不用科学计数法，避免出现 1e-5 这种字样
        ax2.ticklabel_format(style='plain', axis='y')
    
        # ===== 小号图例，放在曲线框右上角内部 =====
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper right',           # 右上角
            bbox_to_anchor=(0.97, 0.97), # 稍微往里一点，避免紧贴边框
            framealpha=0.8,
            facecolor='white',
            edgecolor='black',
            prop={'size': 8},            # 字体小一点
            handlelength=1.2,            # 线段短一点
            borderpad=0.3,               # 内边距小一点
            labelspacing=0.25            # 行间距小一点
        )
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)






