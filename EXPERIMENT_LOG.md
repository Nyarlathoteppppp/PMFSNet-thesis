## 2026-03-19
branch: dev
model: PMFSNet
dataset: ISIC-2018
python train.py --dataset ISIC-2018 --model MSA_MobileNetV2_UNet --dimension 2d --scaling_version BASIC --epoch 150
[2026-03-20 03:08:31]  epoch:[00149/00149]  lr:0.000020  train_loss:0.045536  train_DSC:0.911538  train_IoU:0.885636  train_ACC:0.963893  train_JI:0.852917  valid_DSC:0.880454  valid_IoU:0.814348  valid_ACC:0.956256  valid_JI:0.808035  best_JI:0.808156
改了一下训练速度，然后测试了基线模型