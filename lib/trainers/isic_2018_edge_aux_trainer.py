import os
import datetime

import numpy as np
import torch
import torch.nn.functional as F

from lib import utils
from .isic_2018_trainer import ISIC2018Trainer


class ISIC2018EdgeAuxTrainer(ISIC2018Trainer):
    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):
        super().__init__(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
        self.edge_loss_weight = float(os.environ.get("XBW_EDGE_LOSS_WEIGHT", opt.get("edge_loss_weight", 0.1)))
        self.edge_kernel_size = opt.get("edge_kernel_size", 3)
        self.edge_loss_function = torch.nn.BCEWithLogitsLoss()

    def build_boundary_target(self, target):
        foreground = (target == 1).float().unsqueeze(1)
        kernel_size = self.edge_kernel_size
        padding = kernel_size // 2
        dilated = F.max_pool2d(foreground, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-foreground, kernel_size=kernel_size, stride=1, padding=padding)
        return (dilated - eroded).clamp_(0.0, 1.0)

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):
            input_tensor = input_tensor.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            seg_output, edge_output = self.model(input_tensor)
            seg_loss = self.loss_function(seg_output, target)
            edge_target = self.build_boundary_target(target)
            edge_loss = self.edge_loss_function(edge_output, edge_target)
            total_loss = seg_loss + self.edge_loss_weight * edge_loss

            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.calculate_metric_and_update_statistcs(seg_output, target, len(target), total_loss, mode="train")

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
                train_class_IoU = np.nan_to_num(train_class_IoU)
                print_str = (
                    "[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch, self.end_epoch - 1,
                            batch_idx + 1, len(self.train_data_loader),
                            self.optimizer.param_groups[0]['lr'],
                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                            train_class_IoU[1],
                            self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]))
                print(print_str)
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt(print_str, self.log_txt_path)
