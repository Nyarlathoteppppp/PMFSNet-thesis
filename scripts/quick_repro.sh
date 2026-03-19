#!/usr/bin/env bash
set -euo pipefail

echo "--- PMFSNet 快速复现脚本（帮助打印 & 检查）---"
echo "注意：先激活你的 conda 环境：conda activate /root/autodl-tmp/xbw/envs/pmfs"

if ! command -v python >/dev/null 2>&1; then
  echo "找不到 python，请先激活 conda 环境或安装 Python。"
  exit 1
fi

echo "Python: $(python -V 2>&1)"

if [ ! -f requirements.txt ]; then
  echo "警告：当前目录缺少 requirements.txt —— 请确认在 PMFSNet 项目根目录运行此脚本。"
fi

echo
echo "示例：检查权重文件类型（.pth vs .state）"
echo "python - <<'PY'"
echo "import torch,sys"
echo "obj=torch.load('runs/.../checkpoint.state', map_location='cpu')"
echo "print(type(obj))"
echo "print(list(obj.keys())[:10] if isinstance(obj, dict) else 'not dict')"
echo "PY"

echo
echo "示例：运行一次推理（请替换权重路径和图片路径）"
echo "python inference.py --dataset ISIC-2018 --model MSA_MobileNetV2_UNet --pretrain_weight runs/.../best_*.pth --dimension 2d --scaling_version BASIC --image_path ./images/xxx.jpg"

echo
echo "示例：可视化掩码（执行后会生成 _overlay.jpg）"
echo "python - <<'PY'"
echo "import cv2"
echo "img='datasets/ISIC-2018/test/images/ISIC_0000003.jpg'"
echo "mask='datasets/ISIC-2018/test/images/ISIC_0000003_segmentation.jpg'"
echo "I=cv2.imread(img)"
echo "M=cv2.imread(mask, cv2.IMREAD_UNCHANGED)"
echo "M=cv2.resize(M,(I.shape[1],I.shape[0]),interpolation=cv2.INTER_NEAREST)"
echo "I[M>0]=[0,0,255]"
echo "cv2.imwrite(img.replace('.jpg','_overlay.jpg'),I)"
echo "print('saved', img.replace('.jpg','_overlay.jpg'))"
echo "PY"

echo
echo "脚本结束：上面命令均为示例（只打印），如需我替你执行具体命令，请告诉我要运行的权重文件路径与任务（推理/测试/训练）。"
