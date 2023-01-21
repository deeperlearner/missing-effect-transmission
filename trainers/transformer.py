import os
import time

import torch
from torchvision.utils import make_grid
import numpy as np
import pandas as pd

from base import BaseTrainer
from models.metric import MetricTracker
from utils import inf_loop, consuming_time


class Pretrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, torch_objs: dict, save_dir, resume, device, **kwargs):
        self.device = device
        super(Pretrainer, self).__init__(torch_objs, save_dir, **kwargs)

        if resume is not None:
            self._resume_checkpoint(resume)

        # data_loaders
        self.do_validation = self.valid_data_loaders["data"] is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loaders["data"])
        else:
            # iteration-based training
            self.train_data_loaders["data"] = inf_loop(self.train_data_loaders["data"])
        self.log_step = int(np.sqrt(self.train_data_loaders["data"].batch_size))
        self.train_step, self.valid_step = 0, 0

        # models
        self.transformer = self.models["transformer"]

        # losses
        self.recon_loss = self.losses["recon_loss"]

        # metrics
        keys_loss = ["recon_loss"]
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = []  # pretrainer do not compute these metrics
        self.train_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )

        # optimizers
        self.optimizer = self.optimizers["transformer"]

        # learning rate schedulers
        self.do_lr_scheduling = len(self.lr_schedulers) > 0
        self.lr_scheduler = self.lr_schedulers["transformer"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.transformer.train()
        self.train_metrics.reset()
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

        train_loader = self.train_data_loaders["data"]
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: (x_num, x_cat, x_num_mask, x_cat_mask)
            # target: (day_delta, group)
            data = [x.to(self.device) for x in data]
            event_time, target = [y.to(self.device) for y in target]

            self.optimizer.zero_grad()
            output = self.transformer(data)
            # print(output.size())
            # print(target.size())
            # print(output)
            # os._exit(0)
            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))
            loss = self.recon_loss(output, data)
            if self.apex:
                with self.amp.scale_loss(loss, self.optimizer) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss.backward()
            self.optimizer.step()

            self.train_step += 1
            self.writer.set_step(self.train_step)
            self.train_metrics.iter_update("recon_loss", loss.item())

            # for met in self.metrics_iter:
            #     self.train_metrics.iter_update(met.__name__, met(target, output))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(
                    f"{key}: {value:.6f}" for key, value in current_metrics.items()
                )
                self.logger.debug(epoch_debug + metrics_debug)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        end = time.time()

        # for met in self.metrics_epoch:
        #     self.train_metrics.epoch_update(met.__name__, met(targets, outputs))

        train_log = self.train_metrics.result()

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index("val_" + valid_log.index.astype(str), inplace=True)

        if self.do_lr_scheduling:
            self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        epoch_log = {
            "epochs": epoch,
            "iterations": self.len_epoch * epoch,
            "Runtime": consuming_time(start, end),
        }
        epoch_info = ", ".join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.transformer.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)

            valid_loader = self.valid_data_loaders["data"]
            for batch_idx, (data, target) in enumerate(valid_loader):
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                data = [x.to(self.device) for x in data]
                # x_mask = self.model.generate_square_subsequent_mask(x_mask)
                event_time, target = [y.to(self.device) for y in target]

                self.optimizer.zero_grad()
                output = self.transformer(data)
                loss = self.recon_loss(output, data)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                self.valid_step += 1
                self.writer.set_step(self.valid_step, "valid")
                self.valid_metrics.iter_update("recon_loss", loss.item())
                # for met in self.metrics_iter:
                #     self.valid_metrics.iter_update(met.__name__, met(target, output))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # for met in self.metrics_epoch:
            #     self.valid_metrics.epoch_update(met.__name__, met(targets, outputs))

            # if self.metrics_threshold is not None:
            #     self.threshold = self.metrics_threshold(targets, outputs)

        # # add histogram of model parameters to the tensorboard
        # for name, param in self.models['model'].named_parameters():
        #     self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log

    def _progress(self, batch_idx):
        ratio = "[{}/{} ({:.0f}%)]"
        return ratio.format(
            batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch
        )

    def get_transformer(self):
        return self.transformer


class Finetuner(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self, torch_objs: dict, save_dir, resume, device, **kwargs
    ):
        self.device = device
        super(Finetuner, self).__init__(torch_objs, save_dir, **kwargs)

        if resume is not None:
            self._resume_checkpoint(resume)

        # data_loaders
        self.do_validation = self.valid_data_loaders["data"] is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loaders["data"])
        else:
            # iteration-based training
            self.train_data_loaders["data"] = inf_loop(self.train_data_loaders["data"])
        self.log_step = int(np.sqrt(self.train_data_loaders["data"].batch_size))
        self.train_step, self.valid_step = 0, 0

        # models
        self.finetune_enc = self.models["finetune_enc"]

        # losses
        self.focal_loss = self.losses["focal_loss"]

        # metrics
        keys_loss = ["focal_loss"]
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        self.train_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )

        # optimizers
        self.optimizer = self.optimizers["finetune_enc"]

        # learning rate schedulers
        self.do_lr_scheduling = len(self.lr_schedulers) > 0
        self.lr_scheduler = self.lr_schedulers["finetune_enc"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.finetune_enc.train()
        self.train_metrics.reset()
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            event_times = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

        train_loader = self.train_data_loaders["data"]
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: (x_num, x_cat, x_num_mask, x_cat_mask)
            # target: (day_delta, group)
            data = [x.to(self.device) for x in data]
            event_time, target = [y.to(self.device) for y in target]

            self.optimizer.zero_grad()
            output = self.finetune_enc(data)
            # print(output.size())
            # print(target.size())
            # os._exit(0)
            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output))
                event_times = torch.cat((event_times, event_time))
                targets = torch.cat((targets, target))
            loss = self.focal_loss(output, target)
            if self.apex:
                with self.amp.scale_loss(loss, self.optimizer) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss.backward()
            self.optimizer.step()

            self.train_step += 1
            self.writer.set_step(self.train_step)
            self.train_metrics.iter_update("focal_loss", loss.item())

            for met in self.metrics_iter:
                self.train_metrics.iter_update(met.__name__, met(target, output))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(
                    f"{key}: {value:.6f}" for key, value in current_metrics.items()
                )
                self.logger.debug(epoch_debug + metrics_debug)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        end = time.time()

        for met in self.metrics_epoch:
            self.train_metrics.epoch_update(met.__name__, met(event_times, targets, outputs))

        train_log = self.train_metrics.result()

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index("val_" + valid_log.index.astype(str), inplace=True)

        if self.do_lr_scheduling:
            self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        epoch_log = {
            "epochs": epoch,
            "iterations": self.len_epoch * epoch,
            "Runtime": consuming_time(start, end),
        }
        epoch_info = ", ".join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.finetune_enc.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                event_times = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)

            valid_loader = self.valid_data_loaders["data"]
            for batch_idx, (data, target) in enumerate(valid_loader):
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                data = [x.to(self.device) for x in data]
                event_time, target = [y.to(self.device) for y in target]

                output = self.finetune_enc(data)
                loss = self.focal_loss(output, target)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    event_times = torch.cat((event_times, event_time))
                    targets = torch.cat((targets, target))

                self.valid_step += 1
                self.writer.set_step(self.valid_step, "valid")
                self.valid_metrics.iter_update("focal_loss", loss.item())
                for met in self.metrics_iter:
                    self.valid_metrics.iter_update(met.__name__, met(target, output))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            for met in self.metrics_epoch:
                self.valid_metrics.epoch_update(met.__name__, met(event_times, targets, outputs))

            if self.metrics_threshold is not None:
                self.threshold = self.metrics_threshold(targets, outputs)

        # # add histogram of model parameters to the tensorboard
        # for name, param in self.models['model'].named_parameters():
        #     self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log

    def _progress(self, batch_idx):
        ratio = "[{}/{} ({:.0f}%)]"
        return ratio.format(
            batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch
        )
