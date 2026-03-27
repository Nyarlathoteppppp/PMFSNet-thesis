# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib import models


ISIC_OPT = {
    "dataset_name": "ISIC-2018",
    "dataset_path": str(PROJECT_ROOT / "datasets" / "ISIC-2018"),
    "resize_shape": (224, 224),
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "cuda": True,
}

MODEL_CONFIGS = [
    {
        "name": "PMFSNet",
        "checkpoint": PROJECT_ROOT / "runs" / "2026-03-20-17-05-10_PMFSNet_ISIC-2018" / "checkpoints" / "best_PMFSNet.pth",
    },
    {
        "name": "UNet",
        "checkpoint": PROJECT_ROOT / "runs" / "2026-03-21-15-49-13_UNet_ISIC-2018" / "checkpoints" / "best_UNet.pth",
    },
    {
        "name": "xbwPMFSNet",
        "checkpoint": PROJECT_ROOT / "runs" / "2026-03-23-22-00-44_xbwPMFSNet_ISIC-2018" / "checkpoints" / "best_xbwPMFSNet.pth",
    },
    {
        "name": "xbwPMFSNetV2",
        "checkpoint": PROJECT_ROOT / "runs" / "2026-03-24-14-19-03_xbwPMFSNetV2_ISIC-2018" / "checkpoints" / "best_xbwPMFSNetV2.pth",
    },
    {
        "name": "xbwPMFSNetEdgeAux",
        "checkpoint": PROJECT_ROOT / "runs" / "2026-03-24-23-25-18_xbwPMFSNetEdgeAux_ISIC-2018" / "checkpoints" / "best_xbwPMFSNetEdgeAux.pth",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15, help="number of sorted ISIC test images to export")
    parser.add_argument("--output-tag", type=str, default=None, help="optional custom output directory suffix")
    return parser.parse_args()


def build_opt(model_name: str, device: torch.device) -> dict:
    opt = dict(ISIC_OPT)
    opt["model_name"] = model_name
    opt["device"] = device
    return opt


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    opt = build_opt(model_name, device)
    model = models.get_model(opt).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_state = model.state_dict()
    load_count = 0
    for key in model_state.keys():
        if key in state and model_state[key].shape == state[key].shape:
            model_state[key].copy_(state[key])
            load_count += 1
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Loaded {model_name}: {load_count}/{len(model_state)} tensors from {checkpoint_path}")
    return model


def list_first_images(limit: int):
    image_dir = PROJECT_ROOT / "datasets" / "ISIC-2018" / "test" / "images"
    names = sorted([p.name for p in image_dir.iterdir() if p.is_file()])[:limit]
    return [image_dir / name for name in names]


_TRANSFORM = transforms.Compose([
    transforms.Resize(ISIC_OPT["resize_shape"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=ISIC_OPT["normalize_means"], std=ISIC_OPT["normalize_stds"]),
])


def load_image_tensor(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(image)
    return image, tensor


def load_gt_mask(image_path: Path):
    stem = image_path.stem
    gt_path = PROJECT_ROOT / "datasets" / "ISIC-2018" / "test" / "annotations" / f"{stem}_segmentation.png"
    mask = Image.open(gt_path).convert("L")
    mask_np = (np.array(mask) > 0).astype(np.uint8)
    return gt_path, mask_np


def predict_mask(model, image_tensor: torch.Tensor, original_size):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(next(model.parameters()).device))
        if isinstance(output, (tuple, list)):
            output = output[0]
        pred = torch.argmax(output, dim=1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()
    pred_img = Image.fromarray(pred * 255)
    pred_img = pred_img.resize(original_size, resample=Image.Resampling.NEAREST)
    return (np.array(pred_img) > 0).astype(np.uint8)


def overlay_mask(image_np: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    overlay = image_np.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def overlay_prediction_vs_gt(image_np: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.5):
    overlay = image_np.astype(np.float32).copy()
    overlap = pred_mask.astype(bool) & gt_mask.astype(bool)
    mismatch = pred_mask.astype(bool) ^ gt_mask.astype(bool)
    green = np.array((0, 255, 0), dtype=np.float32)
    red = np.array((255, 0, 0), dtype=np.float32)
    overlay[overlap] = (1.0 - alpha) * overlay[overlap] + alpha * green
    overlay[mismatch] = (1.0 - alpha) * overlay[mismatch] + alpha * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_panel(image_name: str, original: Image.Image, gt_mask: np.ndarray, pred_masks: dict, output_path: Path):
    original_np = np.array(original)
    cols = [
        ("Original", original_np),
        ("GT", overlay_mask(original_np, gt_mask, (0, 255, 0))),
    ]
    for model_name, mask in pred_masks.items():
        cols.append((model_name, overlay_prediction_vs_gt(original_np, mask, gt_mask)))

    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6), dpi=180)
    for ax, (title, img) in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(image_name, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = list_first_images(limit=args.limit)
    tag = args.output_tag or f"2026-03-25_isic_first{args.limit}_model_compare"
    output_root = PROJECT_ROOT / "runs" / "qualitative" / tag
    panel_dir = output_root / "panels"
    raw_dir = output_root / "raw_masks"
    original_dir = output_root / "originals"
    gt_dir = output_root / "ground_truth"
    output_root.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "images": [p.name for p in image_paths],
        "models": {cfg["name"]: str(cfg["checkpoint"]) for cfg in MODEL_CONFIGS},
        "limit": args.limit,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    originals = {}
    gt_masks = {}
    for image_path in image_paths:
        image, _ = load_image_tensor(image_path)
        gt_path, gt_mask = load_gt_mask(image_path)
        originals[image_path.name] = image
        gt_masks[image_path.name] = gt_mask
        image.save(original_dir / image_path.name)
        Image.open(gt_path).save(gt_dir / gt_path.name)

    predictions = {cfg["name"]: {} for cfg in MODEL_CONFIGS}
    for cfg in MODEL_CONFIGS:
        model = load_model(cfg["name"], cfg["checkpoint"], device)
        for image_path in image_paths:
            _, tensor = load_image_tensor(image_path)
            pred_mask = predict_mask(model, tensor, originals[image_path.name].size)
            predictions[cfg["name"]][image_path.name] = pred_mask
            save_mask(pred_mask, raw_dir / cfg["name"] / f"{image_path.stem}_pred.png")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for image_path in image_paths:
        pred_masks = {cfg["name"]: predictions[cfg["name"]][image_path.name] for cfg in MODEL_CONFIGS}
        save_panel(
            image_name=image_path.name,
            original=originals[image_path.name],
            gt_mask=gt_masks[image_path.name],
            pred_masks=pred_masks,
            output_path=panel_dir / f"{image_path.stem}_panel.png",
        )

    print(f"Saved qualitative comparison outputs to: {output_root}")
    print("Images:")
    for image_path in image_paths:
        print(f"  - {image_path.name}")


if __name__ == "__main__":
    main()
