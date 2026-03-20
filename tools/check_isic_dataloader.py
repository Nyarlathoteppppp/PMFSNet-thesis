times = []
import os
import time
import random
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleISICDataset(Dataset):
    def __init__(self, root, mode='train', resize=(224, 224), heavy_aug=True):
        self.root = root
        self.mode = mode
        self.resize = resize
        self.heavy_aug = heavy_aug
        img_dir = os.path.join(root, 'train' if mode == 'train' else 'test', 'images')
        lbl_dir = os.path.join(root, 'train' if mode == 'train' else 'test', 'annotations')
        self.images = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.lower().endswith('.jpg') and '_segmentation' not in p])
        self.labels = sorted([os.path.join(lbl_dir, p) for p in os.listdir(lbl_dir) if p.lower().endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lbl = Image.open(self.labels[idx]).convert('L')
        # simple pipeline to mimic project transforms
        if self.heavy_aug and self.mode == 'train':
            # random resized crop
            w, h = img.size
            scale = random.uniform(0.4, 1.0)
            tw = int(self.resize[0] * scale)
            th = int(self.resize[1] * scale)
            if w > tw and h > th:
                left = random.randint(0, w - tw)
                top = random.randint(0, h - th)
                img = img.crop((left, top, left + tw, top + th)).resize(self.resize)
            else:
                img = img.resize(self.resize)
            # color jitter
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
            # gaussian noise
            arr = np.array(img).astype(np.float32) / 255.0
            if random.random() < 0.2:
                arr += np.random.normal(0, 0.01, arr.shape)
            # random flip
            if random.random() < 0.5:
                arr = arr[:, ::-1, :]
            if random.random() < 0.5:
                arr = arr[::-1, :, :]
            img = (arr * 255).astype(np.uint8)
        else:
            img = img.resize(self.resize)
            img = np.array(img)

        # to tensor
        img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().div(255)
        lbl = torch.from_numpy(np.array(lbl)).float()
        return img, lbl


def main():
    root = os.path.join('.', 'datasets', 'ISIC-2018')
    batch_size = 32
    num_workers = 8
    print('bench params:', {'root': root, 'batch_size': batch_size, 'num_workers': num_workers})
    ds = SimpleISICDataset(root, mode='train', resize=(224, 224), heavy_aug=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    print('loader batches:', len(dl))
    it = iter(dl)
    warm = 2
    n = 10
    for i in range(warm):
        try:
            _ = next(it)
        except StopIteration:
            break
    times = []
    for i in range(n):
        t0 = time.time()
        try:
            _ = next(it)
        except StopIteration:
            break
        t1 = time.time()
        times.append(t1 - t0)
        print(f'batch {i} load time: {t1 - t0:.4f}s')
    if times:
        print('avg load time:', sum(times)/len(times))
        print('min:', min(times), 'max:', max(times))


if __name__ == '__main__':
    main()
