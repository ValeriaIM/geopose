import dataclasses
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from timm.utils import AverageMeter
from torch.nn import DataParallel, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training import losses
from training.config import load_config
from training.losses import LossCalculator
from training.sampler import DistributedWeightedRandomSampler
from training.utils import create_optimizer
from utilities import unet


@dataclasses.dataclass
class TrainConfiguration:
    config_path: str
    gpu: str = "0",
    distributed: bool = False
    from_zero: bool = False
    zero_score: bool = False
    local_rank: int = 0,
    freeze_epochs: int = 0,
    test_every: int = 1
    world_size: int = 1
    output_dir: str = "/home/s0105/_scratch2/project/weights/"
    prefix: str = ""
    resume_checkpoint: str = None
    workers: int = 8
    log_dir: str = "logs"


class Evaluator(ABC):
    @abstractmethod
    def init_metrics(self) -> Dict:
        pass

    @abstractmethod
    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False,
                 local_rank: int = 0, snapshot_name: str = "") -> Dict:
        pass

    @abstractmethod
    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        pass


class LossFunction:

    def __init__(self, loss: LossCalculator, name: str, weight: float = 1, display: bool = False):
        super().__init__()
        self.loss = loss
        self.name = name
        self.weight = weight
        self.display = display


class PytorchTrainer(ABC):
    def __init__(self, train_config: TrainConfiguration, evaluator: Evaluator,
                 fold: int,
                 train_data: Dataset,
                 val_data: Dataset) -> None:
        print("init pytorch trainer")
        super().__init__()
        self.fold = fold
        self.train_config = train_config
        self.conf = load_config(train_config.config_path)
        self._init_distributed()
        self.evaluator = evaluator
        self.current_metrics = evaluator.init_metrics()
        self.current_epoch = 0
        self.model = self._init_model()
        self.losses = self._init_loss_functions()
        self.optimizer, self.scheduler = create_optimizer(self.conf['optimizer'], self.model)
        self._init_amp()
        self.train_data = train_data
        self.val_data = val_data

        self.summary_writer = SummaryWriter(os.path.join(train_config.log_dir,  self.snapshot_name))

    def fit(self):
        print("fit pytorch trainer")
        for epoch in range(self.current_epoch, self.conf["optimizer"]["schedule"]["epochs"]):
            self.current_epoch = epoch
            self.model.train()
            self._freeze()
            self._run_one_epoch_train(self.get_train_loader())
            self.model.eval()
            if self.train_config.local_rank == 0:
                self._save_last()
            if (self.current_epoch + 1) % self.train_config.test_every == 0:
                metrics = self.evaluator.validate(self.get_val_loader(), self.model,
                                                  distributed=self.train_config.distributed,
                                                  local_rank=self.train_config.local_rank,
                                                  snapshot_name=self.snapshot_name)
                if self.train_config.local_rank == 0:
                    improved_metrics = self.evaluator.get_improved_metrics(self.current_metrics, metrics)
                    self.current_metrics.update(improved_metrics)
                    self._save_best(improved_metrics)
                    for k, v in metrics.items():
                        self.summary_writer.add_scalar('val/{}'.format(k), float(v), global_step=self.current_epoch)

    def _save_last(self):
        self.model = self.model.eval()
        torch.save({
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'metrics': self.current_metrics,

        }, os.path.join(self.train_config.output_dir, self.snapshot_name + "_last"))

    def _save_best(self, improved_metrics: Dict):
        print("__save_best pytorch trainer")
        self.model = self.model.eval()
        for metric_name in improved_metrics.keys():
            torch.save({
                'epoch': self.current_epoch,
                'state_dict': self.model.state_dict(),
                'metrics': self.current_metrics,

            }, os.path.join(self.train_config.output_dir, self.snapshot_name + "_" + metric_name))

    def _run_one_epoch_train(self, loader: DataLoader):
        iterator = tqdm(loader)
        loss_meter = AverageMeter()
        avg_meters = {"loss": loss_meter}
        for loss_def in self.losses:
            if loss_def.display:
                avg_meters[loss_def.name] = AverageMeter()

        if self.conf["optimizer"]["schedule"]["mode"] == "epoch":
            self.scheduler.step(self.current_epoch)
        for i, sample in enumerate(iterator):
            # todo: make configurable
            imgs = sample["image"].cuda().float()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(imgs)
                # print(f"output: {output.keys()}")
                # print(f"height size: {output['height'].size()}, mag size: {output['mag'].size()}")
                total_loss = 0
                for loss_def in self.losses:
                    l = loss_def.loss.calculate_loss(output, sample)
                    if loss_def.display:
                        avg_meters[loss_def.name].update(l.item(), imgs.size(0))
                    total_loss += loss_def.weight * l

            loss_meter.update(total_loss.item(), imgs.size(0))
            avg_metrics = {k: v.avg for k, v in avg_meters.items()}
            iterator.set_postfix({"lr": float(self.scheduler.get_lr()[-1]),
                                  "epoch": self.current_epoch,
                                  **avg_metrics
                                  })
            self.gscaler.scale(total_loss).backward()
            self.gscaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.gscaler.step(self.optimizer)
            self.gscaler.update()
            torch.cuda.synchronize()
            dist.barrier()
            if self.conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
                self.scheduler.step(i + self.current_epoch * len(loader))
        if self.train_config.local_rank == 0:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                self.summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=self.current_epoch)
            self.summary_writer.add_scalar('train/loss', float(loss_meter.avg), global_step=self.current_epoch)

    @property
    def train_batch_size(self):
        return self.conf["optimizer"]["train_bs"]

    @property
    def val_batch_size(self):
        return self.conf["optimizer"]["val_bs"]

    def get_train_loader(self) -> DataLoader:
        train_sampler = None
        if self.train_config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
            if hasattr(self.train_data, "get_weights"):
                train_sampler = DistributedWeightedRandomSampler(self.train_data, self.train_data.get_weights())
            train_sampler.set_epoch(self.current_epoch)
        train_data_loader = DataLoader(self.train_data, batch_size=self.train_batch_size,
                                       num_workers=self.train_config.workers,
                                       shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                       drop_last=True)
        print(f'train_data_loader len: {len(train_data_loader)}, batch size: {self.train_batch_size}')

        return train_data_loader

    def get_val_loader(self) -> DataLoader:
        val_sampler = None
        if self.train_config.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_data, shuffle=False)
        val_data_loader = DataLoader(self.val_data, sampler=val_sampler, batch_size=self.val_batch_size,
                                     num_workers=self.train_config.workers,
                                     shuffle=False,
                                     pin_memory=False)
        return val_data_loader

    @property
    def snapshot_name(self):
        return "{}{}_{}_{}".format(self.train_config.prefix, self.conf["network"],
                                   self.conf["encoder_params"]["encoder"], self.fold)

    def _freeze(self):
        if hasattr(self.model.module, "encoder"):
            encoder = self.model.module.encoder
        elif hasattr(self.model.module, "encoder_stages"):
            encoder = self.model.module.encoder_stages
        else:
            logging.warn("unknown encoder model")
            return
        if self.current_epoch < self.train_config.freeze_epochs:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = True

    def _init_amp(self):
        self.gscaler = torch.cuda.amp.GradScaler()

        if self.train_config.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.train_config.local_rank],
                                                 output_device=self.train_config.local_rank,
                                                 find_unused_parameters=True)
        else:
            self.model = DataParallel(self.model).cuda()

    def _init_distributed(self):
        if self.train_config.distributed:
            self.pg = dist.init_process_group(backend="nccl",
                                              rank=self.train_config.local_rank,
                                              world_size=self.train_config.world_size)

            torch.cuda.set_device(self.train_config.local_rank)
        else:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = self.train_config.gpu

    def _load_checkpoint(self, model: torch.nn.Module):
        checkpoint_path = self.train_config.resume_checkpoint
        if not checkpoint_path:
            return
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
                orig_state_dict = model.state_dict()
                mismatched_keys = []
                for k, v in state_dict.items():
                    ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
                    if v.size() != ori_size:
                        print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                        mismatched_keys.append(k)
                for k in mismatched_keys:
                    del state_dict[k]
                model.load_state_dict(state_dict, strict=False)
                if not self.train_config.from_zero:
                    self.current_epoch = checkpoint['epoch']
                    if not self.train_config.zero_score:
                        self.current_metrics = checkpoint.get('metrics', self.evaluator.init_metrics())
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_path, checkpoint['epoch']))
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
        if self.train_config.from_zero:
            self.current_metrics = self.evaluator.init_metrics()
            self.current_epoch = 0

    def _init_model(self):
        print(self.train_config)

        model = unet.__dict__[self.conf['network']](**self.conf["encoder_params"])
        model = model.cuda()
        self._load_checkpoint(model)
        model.add_segm_head()
        # added segm_head to cuda
        model = model.cuda()

        if self.train_config.distributed:
            model = SyncBatchNorm.convert_sync_batchnorm(model, self.pg)
        model = model.to(memory_format=torch.channels_last)
        return model

    def _init_loss_functions(self) -> List[LossFunction]:
        assert self.conf['losses']
        loss_functions = []
        for loss_def in self.conf['losses']:
            loss_fn = losses.__dict__[loss_def["type"]](**loss_def["params"])
            loss_weight = loss_def["weight"]
            display = loss_def["display"]
            loss_functions.append(LossFunction(loss_fn, loss_def["name"], loss_weight, display))

        return loss_functions
