# PMFSNet — 快速复现说明（精简）

注意：这是为你的服务器上的 `xbw` 目录准备的复现笔记，仅在该目录下创建/修改文件，不会影响其他用户文件。

1. 环境（你已准备好）

- 已知 conda 环境路径（示例）：

  ```bash
  conda activate /root/autodl-tmp/xbw/envs/pmfs
  ```

- 若尚未安装依赖：

  ```bash
  pip install -r requirements.txt
  ```

2. 检查 GPU / PyTorch

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

3. 常用运行命令（复制并替换路径）

- 训练：
```bash
python train.py --dataset ISIC-2018 --model PMFSNet --dimension 2d --scaling_version BASIC --epoch 150
```

- 测试：
```bash
python test.py --dataset ISIC-2018 --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-BASIC_ISIC2018.pth --dimension 2d --scaling_version BASIC
```

- 单张图像推理（务必使用 `.pth` 权重，而非 `.state`）：
```bash
python inference.py --dataset ISIC-2018 --model MSA_MobileNetV2_UNet --pretrain_weight runs/.../best_MSA_MobileNetV2_UNet.pth --dimension 2d --scaling_version BASIC --image_path ./images/xxx.jpg
```

4. 常见问题快速排查

- 权重类型：不要把包含优化器/调度器信息的 `.state` 文件传给 `--pretrain_weight`，应使用纯模型权重 `.pth`。检查文件内容：

```bash
python - <<'PY'
import torch,sys
obj=torch.load(sys.argv[1], map_location='cpu')
print(type(obj))
if isinstance(obj, dict):
    print('keys:', list(obj.keys())[:10])
PY
# 用法： python check_weight.py runs/.../checkpoint.state
```

- 掩码可视化：`lib/testers/isic_2018_tester.py` 保存的是 0/1 掩码，直接用普通查看器会显得很暗，若控制台 `np.sum(segmented_image > 0)` 非零，可按下面方式可视化：

```bash
python - <<'PY'
import cv2
img='datasets/ISIC-2018/test/images/ISIC_0000003.jpg'
mask='datasets/ISIC-2018/test/images/ISIC_0000003_segmentation.jpg'
I=cv2.imread(img)
M=cv2.imread(mask, cv2.IMREAD_UNCHANGED)
M=cv2.resize(M,(I.shape[1],I.shape[0]),interpolation=cv2.INTER_NEAREST)
I[M>0]=[0,0,255]
cv2.imwrite(img.replace('.jpg','_overlay.jpg'),I)
print('saved', img.replace('.jpg','_overlay.jpg'))
PY
```

- BatchNorm 尺寸不匹配：如果出现 `running_mean should contain X elements not Y`，通常是通道数或 `skip_indices`/`scaling_version` 与模型 backbone 不一致，检查你传入的 `--model`、`--scaling_version` 与代码中模型配置是否匹配。

5. 小技巧与安全

- 加载未知来源模型时，注意 `torch.load` 的安全提示（可研究 `weights_only=True` 的使用）。
- 以项目根目录运行脚本，或在运行前确保 `PYTHONPATH` 包含项目根路径：

```bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
```

6. 其它说明

- 日志与 checkpoint 在 `runs/` 目录下；训练实验文件夹内通常包含 `checkpoints/`（`.pth`）与 `.state` 文件（含训练器状态）。
- 我已在 `scripts/quick_repro.sh` 提供一个交互式辅助脚本（仅打印与检查，不会自动修改仓库外的文件）。

如需，我可以把某些命令写成可执行脚本并示范一次本地（快速）推理，但需要你授权使用仓库内的某个 `.pth` 权重文件（或允许我使用 `pretrain/` 中的权重）。
