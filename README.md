# PMFSNetEdgeAux

A lightweight boundary-aware extension of PMFSNet for medical image segmentation.

This repository is built upon the original **PMFSNet** and extends it with a **shallow boundary auxiliary supervision branch** for **skin lesion segmentation on ISIC 2018**. The proposed **PMFSNetEdgeAux** keeps the original PMFSNet backbone and global PMFS block unchanged, while introducing a lightweight edge auxiliary head on the stage-1 skip feature to enhance local boundary representation.

The main idea is simple:  
- **global context** is modeled by the original PMFS block at the bottleneck,  
- **local detail** is strengthened by boundary-aware supervision on the shallow skip feature,  
- and both are fused by the original decoder to produce the final segmentation.

## Highlights

- Built on the original lightweight **PMFSNet** framework
- Adds a **boundary auxiliary head** without modifying the main backbone
- Improves local boundary representation while preserving global context modeling
- Includes training, testing, qualitative visualization, and ablation scripts for **ISIC 2018**
- Designed as a research-oriented extension for lightweight medical image segmentation

## Main Result on ISIC 2018

| Method | DSC | IoU | ACC | JI |
|---|---:|---:|---:|---:|
| UNet | 0.8717 | 0.8010 | 0.9524 | 0.7764 |
| PMFSNet | 0.8749 | 0.8103 | 0.9539 | 0.7815 |
| PMFSNetEdgeAux | **0.8810** | **0.8188** | **0.9558** | **0.7916** |

## Edge Loss Ablation

We further study the influence of the boundary supervision weight `λ`:

| Edge Loss Weight | DSC | IoU | ACC | JI |
|---|---:|---:|---:|---:|
| 0 | 0.8801 | 0.8179 | 0.9565 | 0.7898 |
| 0.05 | 0.8802 | 0.8177 | 0.9563 | 0.7901 |
| 0.08 | 0.8765 | 0.8139 | 0.9556 | 0.7843 |
| 0.10 | **0.8810** | **0.8188** | 0.9558 | **0.7916** |
| 0.20 | 0.8781 | 0.8133 | 0.9552 | 0.7863 |

These results show that boundary supervision is helpful, but an overly large weight may interfere with the main segmentation objective. In our experiments, `λ = 0.1` gives the best balance.

## Project Scope

This repository currently focuses on:
- reproducing **PMFSNet** on **ISIC 2018**
- implementing the proposed **PMFSNetEdgeAux**
- comparing baseline and improved models quantitatively and qualitatively
- supporting thesis-oriented experiments and ablation studies

## Acknowledgement

This work is based on the original [PMFSNet](https://github.com/yykzjh/PMFSNet) project. Thanks to the original authors for releasing their code and research.
