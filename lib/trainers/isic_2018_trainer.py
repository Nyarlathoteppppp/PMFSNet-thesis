import datetime
import os

import nni
import numpy as np
import torch
import torch.optim as optim

from lib import utils


class ISIC2018Trainer:
    """
    Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.best_metrics = ""
        self.device = opt["device"]

        if not self.opt["optimize_params"]:
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"], utils.datestr() + "_" + opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                utils.make_dirs(self.checkpoint_dir)
                utils.make_dirs(self.tensorboard_dir)
            utils.pre_write_txt("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]),
                                self.log_txt_path)

        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_metric = opt["best_metric"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

        self.statistics_dict = self.init_statistics_dict()

    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.reset_statistics_dict()

            self.optimizer.zero_grad()

            self.train_epoch(epoch)

            update_flag = self.valid_epoch(epoch)

            train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
            train_class_IoU = np.nan_to_num(train_class_IoU)
            valid_class_IoU = self.statistics_dict["valid"]["total_area_intersect"] / self.statistics_dict["valid"]["total_area_union"]
            valid_class_IoU = np.nan_to_num(valid_class_IoU)
            valid_dsc = self.statistics_dict["valid"]["DSC_sum"] / self.statistics_dict["valid"]["count"]
            valid_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]
            valid_ACC = self.statistics_dict["valid"]["ACC_sum"] / self.statistics_dict["valid"]["count"]

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(valid_JI)
            else:
                self.lr_scheduler.step()

            print_str = (
                "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC:{:.6f}  train_JI:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f}"
                .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch, self.end_epoch - 1,
                        self.optimizer.param_groups[0]['lr'],
                        self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                        self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                        train_class_IoU[1],
                        self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                        self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"],
                        valid_dsc,
                        valid_class_IoU[1],
                        valid_ACC,
                        valid_JI,
                        self.best_metric))

            if update_flag:
                self.best_metrics = print_str

            print(print_str)
            if not self.opt["optimize_params"]:
                utils.pre_write_txt(print_str, self.log_txt_path)

            if self.opt["optimize_params"]:
                nni.report_intermediate_result(valid_JI)

        if self.opt["optimize_params"]:
            nni.report_final_result(self.best_metric)
        else:
            utils.pre_write_txt("\n" + self.best_metrics, self.log_txt_path)

    def train_epoch(self, epoch):

        self.model.train()
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):

            # use non_blocking=True to overlap host->device copies with computation
            input_tensor = input_tensor.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = self.model(input_tensor)
            dice_loss = self.loss_function(output, target)
            dice_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # compute and accumulate metrics on device, only transfer small summaries to CPU
            self.calculate_metric_and_update_statistcs(output, target, len(target), dice_loss, mode="train")

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
                train_class_IoU = np.nan_to_num(train_class_IoU)
                print("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              epoch, self.end_epoch - 1,
                              batch_idx + 1, len(self.train_data_loader),
                              self.optimizer.param_groups[0]['lr'],
                              self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                              train_class_IoU[1],
                              self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                                        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                epoch, self.end_epoch - 1,
                                                batch_idx + 1, len(self.train_data_loader),
                                                self.optimizer.param_groups[0]['lr'],
                                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                                                train_class_IoU[1],
                                                self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]),
                                        self.log_txt_path)

    def valid_epoch(self, epoch):

        self.model.eval()

        with torch.no_grad():

            for batch_idx, (input_tensor, target) in enumerate(self.valid_data_loader):
                input_tensor = input_tensor.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.model(input_tensor)

                # compute metrics on device and transfer summaries
                self.calculate_metric_and_update_statistcs(output, target, len(target), mode="valid")

            cur_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]

            update_flag = False
            if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0:
                self.save(epoch, cur_JI, self.best_metric, type="normal")
            if not self.opt["optimize_params"]:
                self.save(epoch, cur_JI, self.best_metric, type="latest")
            if cur_JI > self.best_metric:
                update_flag = True
                self.best_metric = cur_JI
                if not self.opt["optimize_params"]:
                    self.save(epoch, cur_JI, self.best_metric, type="best")
            return update_flag

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        # output: (B, C, H, W), target: (B, H, W) on device
        # build mask of present classes in this batch (on device)
        unique_index = torch.unique(target).int()
        mask = torch.zeros(self.opt["classes"], device=target.device, dtype=torch.uint8)
        for index in unique_index:
            mask[index] = 1

        # update counts
        self.statistics_dict[mode]["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i]:
                self.statistics_dict[mode]["class_count"][class_name] += cur_batch_size

        if mode == "train" and loss is not None:
            # keep loss aggregation on CPU scalar
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size

        # predictions
        probs = None
        preds = None
        for metric_name, metric_func in self.metric.items():
            if metric_name == "IoU":
                # metric_func likely returns torch hist tensors; call it and accumulate
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict[mode]["total_area_intersect"] += area_intersect.cpu().numpy()
                self.statistics_dict[mode]["total_area_union"] += area_union.cpu().numpy()
            elif metric_name == "DSC":
                # compute Dice on device without converting whole tensors to numpy
                if probs is None:
                    # use the normalization defined by metric if available
                    try:
                        norm = metric_func.normalization
                        probs = norm(output)
                    except Exception:
                        probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                # compute per-sample dice for foreground class and mean
                # assume binary foreground class index 1
                fore = 1 if self.opt["classes"] > 1 else 1
                pred_fg = (preds == fore).float()
                gt_fg = (target == fore).float()
                inter = (pred_fg * gt_fg).view(pred_fg.size(0), -1).sum(dim=1)
                sums = pred_fg.view(pred_fg.size(0), -1).sum(dim=1) + gt_fg.view(gt_fg.size(0), -1).sum(dim=1)
                dice_per_sample = (2.0 * inter) / (sums + 1e-6)
                batch_mean_DSC = float(dice_per_sample.mean().item())
                self.statistics_dict[mode]["DSC_sum"] += batch_mean_DSC * cur_batch_size
            elif metric_name == "JI":
                # Jaccard = intersection / union
                if preds is None:
                    preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
                pred_fg = (preds == 1).float()
                gt_fg = (target == 1).float()
                inter = (pred_fg * gt_fg).view(pred_fg.size(0), -1).sum(dim=1)
                union = ((pred_fg + gt_fg) > 0).view(pred_fg.size(0), -1).sum(dim=1)
                ji_per_sample = inter / (union + 1e-6)
                batch_mean_JI = float(ji_per_sample.mean().item())
                self.statistics_dict[mode]["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "ACC":
                if preds is None:
                    preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
                acc_per_sample = (preds == target).view(preds.size(0), -1).float().mean(dim=1)
                batch_mean_ACC = float(acc_per_sample.mean().item())
                self.statistics_dict[mode]["ACC_sum"] += batch_mean_ACC * cur_batch_size
            else:
                # fallback: call metric_func and try to aggregate per-class results
                per_class_metric = metric_func(output, target)
                # per_class_metric either torch tensor or numpy; ensure torch
                if isinstance(per_class_metric, torch.Tensor):
                    per = per_class_metric
                else:
                    per = torch.as_tensor(per_class_metric, device=target.device)
                per = per * mask.to(per.device)
                avg = (torch.sum(per) / torch.sum(mask.to(per.device))).item() if torch.sum(mask) > 0 else 0.0
                self.statistics_dict[mode][metric_name]["avg"] += avg * cur_batch_size
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[mode][metric_name][class_name] += float(per[j].cpu().item()) * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            "train": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            },
            "valid": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            }
        }
        statistics_dict["train"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["train"]["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["train"]["JI_sum"] = 0.0
        statistics_dict["valid"]["JI_sum"] = 0.0
        statistics_dict["train"]["ACC_sum"] = 0.0
        statistics_dict["valid"]["ACC_sum"] = 0.0
        statistics_dict["train"]["DSC_sum"] = 0.0
        statistics_dict["valid"]["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name]["avg"] = 0.0
            statistics_dict["valid"][metric_name]["avg"] = 0.0
        statistics_dict["train"]["loss"] = 0.0
        statistics_dict["train"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["valid"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        for phase in ["train", "valid"]:
            self.statistics_dict[phase]["count"] = 0
            self.statistics_dict[phase]["total_area_intersect"] = np.zeros((self.opt["classes"],))
            self.statistics_dict[phase]["total_area_union"] = np.zeros((self.opt["classes"],))
            self.statistics_dict[phase]["JI_sum"] = 0.0
            self.statistics_dict[phase]["ACC_sum"] = 0.0
            self.statistics_dict[phase]["DSC_sum"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[phase]["class_count"][class_name] = 0
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name]["avg"] = 0.0
                for _, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[phase][metric_name][class_name] = 0.0

    def save(self, epoch, metric, best_metric, type="normal"):
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        if self.opt["resume"] is not None:
            if self.opt["pretrain"] is None:
                raise RuntimeError("Training weights must be specified to continue training")

            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.start_epoch = resume_state_dict["epoch"] + 1
            self.best_metric = resume_state_dict["best_metric"]
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            model_state_dict = self.model.state_dict()
            load_count = 0
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            self.model.load_state_dict(model_state_dict, strict=True)
            print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
        else:
            if self.opt["pretrain"] is not None:
                pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                model_state_dict = self.model.state_dict()
                load_count = 0
                for param_name in model_state_dict.keys():
                    if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                        model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                        load_count += 1
                self.model.load_state_dict(model_state_dict, strict=True)
                print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
