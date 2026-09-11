# BXMNet 训练参数整理

以下是当前仓库中用于主实验/公平对比的统一训练协议。模型结构快照以各实验目录中的配置 JSON 为准；这里记录的是可复用的公共参数。

## 训练与数据

| 参数 | 固定值 |
|---|---|
| nnU-Net configuration | `3d_fullres` |
| plans | `nnUNetPlans` |
| patch size | `96 × 96 × 96` |
| batch size | `1` |
| epochs | `500` |
| iterations / epoch | `250` |
| validation iterations / epoch | `50` |
| foreground oversampling | `0.33` |
| data augmentation workers | `nnUNet_n_proc_DA=2` |
| AMP | CUDA 下开启 |
| Torch compile | 关闭 |

## 优化器与学习率

| 参数 | 固定值 |
|---|---|
| optimizer | `AdamW` |
| learning rate | `1e-4` |
| betas | `(0.9, 0.999)` |
| weight decay | `1e-4` |
| schedule | `LinearLR warmup → ConstantLR hold → CosineAnnealingLR` |
| warmup | `10` epochs，`start_factor=0.2` |
| hold | `20` epochs |
| cosine `eta_min` | `1e-6` |
| checkpoint interval | 每 `50` epochs |
| validation checkpoint | `checkpoint_best.pth` |

注意：当前 trainer 的实际训练轮数为 500，但 scheduler 配置中的
`total_epochs` 仍是 1000。为复现已有 BXMNet 结果应保留该事实；对比实验中不要只修改一个模型。

另一个需要保留的历史差异是 neck depth：根目录通用 preset 的默认值是 4，
`brats2020_dataset007_5fold_cv` 的有效 fold 配置记录的是 5。集中配置已经按实验分别写入，不要只看公共说明而覆盖它。

## BXMNet 结构快照

- 输入通道：4；输出类别：4（背景、NCR/NET、ED、ET）。
- encoder channels：`[32, 64, 128, 256]`，每 stage 两个 block。
- encoder stride：`[(1,1,1), (2,2,2), (2,2,2), (2,2,2)]`。
- normalization：`InstanceNorm3d(eps=1e-5, affine=True)`。
- activation：`LeakyReLU(inplace=True)`。
- Mamba neck：普通 `nnUNetTrainerBXMNet` 为 `depth=4`；现有 295 例五折最终协议的 `nnUNetTrainerBXMNetRevision` 为 `depth=5`。两者均使用 `full_6` scan、1 个 Mamba layer、drop path `0.1`。
- MAFCM decoder：每 stage 两个 block，deep supervision 开启，trilinear 上采样，`alpha=0.5`、`beta=0.25`。
- OAFF：`groups=4`、`r=8`、`alpha=1.0`、`beta=1.0`、`tau=0.6`，pixel/channel head 均开启。

## 验证与输出

- 区域定义：`WT={1,2,3}`、`TC={1,3}`、`ET={3}`。
- 至少保存 WT/TC/ET Dice、AVG Dice、WT/TC/ET HD95、AVG HD95。
- 保留病例级 CSV，不只保存均值；固定划分文件与病例级结果必须一一对应。
- `export_validation_probabilities=True`；验证默认使用 best checkpoint。

## 已有配置文件的位置

- 五折 BraTS2020 的实际 fold 审计：`brats2020_dataset007_5fold_cv/results/config/fold_*_bxmnet_config.json`；
- BraTS2020 旧 294 例五折运行协议：`brats2020_dataset007_5fold_cv/README.md`；
- BraTS2024 主实验入口：`revision_bxmnet_experiments/BXM-13_brats2024_full/`；
- 根目录通用说明：`BXMNet_training_config_for_baseline_reproduction.md`。
