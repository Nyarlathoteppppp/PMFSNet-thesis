# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import lib.transforms.two as my_transforms


class ISIC2018Dataset(Dataset):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(ISIC2018Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_dir = os.path.join(self.root, "train")
        self.valid_dir = os.path.join(self.root, "test")
        # allow disabling heavy CPU augmentations for faster debugging
        if self.opt.get("no_augment", False):
            self.transforms_dict = {
                "train": my_transforms.Compose([
                    my_transforms.Resize(self.opt["resize_shape"]),
                    my_transforms.ToTensor(),
                    my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
                ]),
                "valid": my_transforms.Compose([
                    my_transforms.Resize(self.opt["resize_shape"]),
                    my_transforms.ToTensor(),
                    my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
                ])
            }
        else:
            self.transforms_dict = {
                "train": my_transforms.Compose([
                    my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                    my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                    my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                    my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                    my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                    my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                    my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                    my_transforms.ToTensor(),
                    my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
                ]),
                "valid": my_transforms.Compose([
                    my_transforms.Resize(self.opt["resize_shape"]),
                    my_transforms.ToTensor(),
                    my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
                ])
            }

        if mode == "train":
            all_images = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.jpg")))
            # filter out files that include segmentation overlays or unexpected names
            self.images_list = [p for p in all_images if "_segmentation" not in os.path.basename(p)]
            all_labels = sorted(glob.glob(os.path.join(self.train_dir, "annotations", "*.png")))
            self.labels_list = all_labels
        else:
            all_images = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.jpg")))
            self.images_list = [p for p in all_images if "_segmentation" not in os.path.basename(p)]
            all_labels = sorted(glob.glob(os.path.join(self.valid_dir, "annotations", "*.png")))
            self.labels_list = all_labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # read image in color and labels in grayscale, convert to RGB
        if cv2 is not None:
            image = cv2.imread(self.images_list[index], cv2.IMREAD_COLOR)
            label = cv2.imread(self.labels_list[index], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read image: {self.images_list[index]}")
            if label is None:
                raise RuntimeError(f"Failed to read label: {self.labels_list[index]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # fallback to PIL if cv2 is unavailable (diagnostic mode)
            img = Image.open(self.images_list[index]).convert('RGB')
            image = np.array(img)
            lbl = Image.open(self.labels_list[index]).convert('L')
            label = np.array(lbl)
        label[label == 255] = 1
        image, label = self.transforms_dict[self.mode](image, label)
        return image, label
