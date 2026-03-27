# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "runs" / "qualitative" / "2026-03-25_isic_first100_model_compare"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "qualitative" / "2026-03-25_isic_paper_figures"

MAIN_STEMS = [
    "ISIC_0000319",
    "ISIC_0000133",
    "ISIC_0000298",
    "ISIC_0000101",
]

SUPP_STEMS = [
    "ISIC_0000323",
    "ISIC_0000235",
]

COLUMN_TITLES = ["Original", "GT", "UNet", "PMFSNet", "PMFSNetEdgeAux"]


def load_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path):
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)


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


def assemble_case(stem: str):
    original = load_rgb(SOURCE_ROOT / "originals" / f"{stem}.jpg")
    gt = load_mask(SOURCE_ROOT / "ground_truth" / f"{stem}_segmentation.png")
    unet = load_mask(SOURCE_ROOT / "raw_masks" / "UNet" / f"{stem}_pred.png")
    pmfs = load_mask(SOURCE_ROOT / "raw_masks" / "PMFSNet" / f"{stem}_pred.png")
    xbw = load_mask(SOURCE_ROOT / "raw_masks" / "xbwPMFSNetEdgeAux" / f"{stem}_pred.png")
    return [
        original,
        overlay_mask(original, gt, (0, 255, 0)),
        overlay_prediction_vs_gt(original, unet, gt),
        overlay_prediction_vs_gt(original, pmfs, gt),
        overlay_prediction_vs_gt(original, xbw, gt),
    ]


def save_grid(stems, output_path: Path, title: str):
    rows = len(stems)
    cols = len(COLUMN_TITLES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows), dpi=220)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for col, col_title in enumerate(COLUMN_TITLES):
        axes[0, col].set_title(col_title, fontsize=12)

    for row, stem in enumerate(stems):
        images = assemble_case(stem)
        for col, img in enumerate(images):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(stem, fontsize=11, rotation=90, labelpad=16)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_single_case_panels(stems, subdir: str):
    target_dir = OUTPUT_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        images = assemble_case(stem)
        fig, axes = plt.subplots(1, len(COLUMN_TITLES), figsize=(4.0 * len(COLUMN_TITLES), 3.8), dpi=220)
        for idx, (ax, img, title) in enumerate(zip(axes, images, COLUMN_TITLES)):
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(stem, fontsize=13)
        fig.tight_layout()
        fig.savefig(target_dir / f"{stem}_paper_panel.png", bbox_inches="tight")
        plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_grid(MAIN_STEMS, OUTPUT_ROOT / "main_figure_grid.png", "Qualitative Comparison on ISIC-2018")
    save_grid(SUPP_STEMS, OUTPUT_ROOT / "supplementary_figure_grid.png", "Supplementary Qualitative Cases")
    save_single_case_panels(MAIN_STEMS, "main_cases")
    save_single_case_panels(SUPP_STEMS, "supp_cases")
    metadata = {
        "source_root": str(SOURCE_ROOT),
        "main_stems": MAIN_STEMS,
        "supp_stems": SUPP_STEMS,
        "columns": COLUMN_TITLES,
    }
    (OUTPUT_ROOT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved paper figures to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
