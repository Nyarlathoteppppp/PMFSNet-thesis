import sys
import time
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from lib.dataloaders import get_dataloader

opt = {
    "dataset_name": "ISIC-2018",
    "dataset_path": os.path.join('.', 'datasets', 'ISIC-2018'),
    "batch_size": 32,
    "num_workers": 8,
    "resize_shape": (224, 224),
    "color_jitter": 0.37,
    "augmentation_p": 0.1,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118286, 0.17080363),
}

print('Opt prepared:', {k: opt[k] for k in ['dataset_name','batch_size','num_workers','resize_shape']})

try:
    train_loader, valid_loader = get_dataloader(opt)
except Exception as e:
    print('get_dataloader error:', e)
    raise

print('train loader len (batches):', len(train_loader))

it = iter(train_loader)

warm = 2
n=10

# warm-up
for i in range(warm):
    try:
        _ = next(it)
    except StopIteration:
        print('iterator exhausted during warmup')
        break

# measure
times = []
for i in range(n):
    t0 = time.time()
    try:
        _ = next(it)
    except StopIteration:
        print('iterator exhausted at iteration', i)
        break
    t1 = time.time()
    times.append(t1-t0)
    print(f'batch {i} load time: {t1-t0:.4f}s')

if times:
    print('avg load time:', sum(times)/len(times))
    print('min:', min(times), 'max:', max(times))
else:
    print('no batches measured')
