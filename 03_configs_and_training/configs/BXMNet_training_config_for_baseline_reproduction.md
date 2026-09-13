# BXMNet 训练配置与 Baseline 公平复现清单

本文档整理当前仓库中可确认的 BXMNet 训练协议，用于复现 SegMamba、SwinUNETR、nnFormer、UNETR、nnU-Net 等其他模型时尽量保持相同实验设置。建议把这里的配置作为主对比的统一协议；模型内部结构可以保留各自官方实现，但数据、split、输入尺寸、训练预算、验证输出和指标计算要尽量一致。

## 1. 推荐统一训练入口

BraTS2024 主结果建议使用修稿实验目录里的入口：

```bash
FOLD=0 sbatch revision_bxmnet_experiments/BXM-13_brats2024_full/scripts/submit_train_fold.sh
```

对应 Python 入口：

```bash
python revision_bxmnet_experiments/BXM-13_brats2024_full/code/train_bxmnet_brats2024.py \
  --fold 0 \
  --dataset Dataset006_BraTS2024_full \
  --configuration 3d_fullres \
  --trainer nnUNetTrainerBXMNet \
  --plans nnUNetPlans \
  --device cuda \
  --num-gpus 1
```

老的根目录入口 `main_training_brats2024.py` 也等价地固定了 `Dataset006_BraTS2024_full`、`3d_fullres`、`nnUNetPlans`、`nnUNetTrainerBXMNet`、`num_gpus=1`、`export_validation_probabilities=True`。

## 2. 数据与预处理协议

| 项目 | BXMNet 当前设置 | Baseline 复现建议 |
|---|---:|---|
| 数据组织 | nnU-Net v2 风格目录 | baseline 尽量读取同一 `nnUNet_raw` / `nnUNet_preprocessed` |
| BraTS2024 数据集名 | `Dataset006_BraTS2024_full` | 完全一致 |
| 配置名 | `3d_fullres` | 完全一致 |
| plans | `nnUNetPlans` | 完全一致，或从同一 plans 导出等价配置 |
| 输入模态数 | 4 | T1/T1ce/T2/FLAIR 顺序必须固定并记录 |
| 输出类别 | 4，含背景 + 3 个肿瘤子区域 | 保持同一标签映射 |
| spacing | 代码 preset 期望 `[1.0, 1.0, 1.0]` | 保持同一重采样 spacing |
| patch size | 代码 preset 期望 `[96, 96, 96]` | 主对比统一使用 96^3 |
| batch size | 代码 preset 期望 `1` | 显存允许也不要随意放大；若必须改，所有模型同步说明 |
| normalization | nnU-Net plans 决定，BraTS 通常 4 通道 Z-score + mask norm | baseline 应复用同一预处理统计 |
| split | `splits_final.json` 优先；不存在时 nnU-Net seed=12345 生成 5-fold | 所有模型必须使用同一 fold/split 文件 |

注意：当前本地工作区只包含 `Dataset001_BraTS2019` 的预处理目录；BraTS2024 的完整 `nnUNetPlans.json` 和 `splits_final.json` 应在集群 `PROJECT_ROOT` 对应的 `nnUNet_preprocessed/Dataset006_BraTS2024_full/` 下确认并归档。

## 3. BXMNet 架构固定项

BXMNet 不是从 plans 动态构建网络，而是在代码里使用冻结 preset：`brats2020_3d_fullres_v1`。这意味着 baseline 对齐时应以数据协议和训练协议为主，不要把 BXMNet 的内部结构强行迁移到其他模型。

| 模块 | 当前 BXMNet 设置 |
|---|---|
| 总体结构 | `DCREncoder -> MambaNeck -> MAFCMResidualDecoder3D` |
| encoder stages | 4 |
| channels | `[32, 64, 128, 256]` |
| kernel sizes | `[(3,3,3)] * 4` |
| strides | `[(1,1,1), (2,2,2), (2,2,2), (2,2,2)]` |
| blocks per encoder stage | `[2, 2, 2, 2]` |
| conv bias | `False` |
| norm | `InstanceNorm3d(eps=1e-5, affine=True)` |
| activation | `LeakyReLU(inplace=True)` |
| DCR stochastic depth | `0.0` |
| DCR squeeze excitation | `False` |
| neck depth | `4` |
| neck drop path | `0.1` |
| HSS scan mode | `full_6` |
| neck mamba layers | `1` |
| decoder blocks per stage | `2` |
| deep supervision | `True` |
| MAFCM alpha / beta | `0.50 / 0.25` |
| upsample mode | `trilinear` |

## 4. 训练超参数

| 项目 | 当前值 |
|---|---:|
| epochs | `500`，由 `nnUNetTrainerBXMNet` 覆盖 |
| iterations per epoch | `250`，继承 nnU-Net trainer |
| validation iterations per epoch | `50`，继承 nnU-Net trainer |
| optimizer | `AdamW` |
| learning rate | `1e-4` |
| betas | `(0.9, 0.999)` |
| weight decay | `1e-4` |
| lr schedule | `LinearLR warmup -> ConstantLR hold -> CosineAnnealingLR` |
| warmup epochs | `10` |
| warmup start factor | `0.2` |
| hold epochs | `20` |
| cosine eta_min | `1e-6` |
| scheduler total_epochs | 代码里写 `1000`，但实际 trainer epochs 为 `500` |
| foreground oversampling | `0.33` |
| probabilistic oversampling | `False` |
| AMP | CUDA 下使用 `autocast` + `GradScaler` |
| torch.compile | 入口脚本通过环境变量关闭 TorchDynamo/TorchInductor/TorchCompile |
| data augmentation workers | `nnUNet_n_proc_DA=2` |
| checkpoint save interval | `50` epochs |
| validation checkpoint | 默认 `val_with_best=True` |
| validation probabilities | `export_validation_probabilities=True` |

容易踩坑的一点：`nnUNetTrainerBXMNet` 的训练轮数是 500，但 `configure_optimizers()` 里的 scheduler `total_epochs` 仍写为 1000。这是当前代码事实。为了完全复现 BXMNet，就保持不变；为了更严格公平对齐 baseline，可以在实验记录里显式写明这一点。

## 5. 数据增强与采样

BXMNet 继承当前 nnU-Net trainer 的 3D fullres augmentation。主要包括：

- 空间变换：rotation 概率 `0.2`，scaling 概率 `0.2`，scale range `(0.7, 1.4)`。
- patch 为各向同性 3D 时，rotation 范围约为 `[-30 deg, 30 deg]`，mirror axes 为 `(0, 1, 2)`。
- Gaussian noise：概率 `0.1`，variance `(0, 0.1)`。
- Gaussian blur：概率 `0.2`，sigma `(0.5, 1.0)`。
- brightness multiplier：概率 `0.15`，range `(0.75, 1.25)`。
- contrast：概率 `0.15`，range `(0.75, 1.25)`。
- simulated low resolution：概率 `0.25`，scale `(0.5, 1)`。
- gamma invert：概率 `0.1`，gamma `(0.7, 1.5)`。
- gamma non-invert：概率 `0.3`，gamma `(0.7, 1.5)`。
- mask normalization：按 plans 的 `use_mask_for_norm` 对应通道执行。
- deep supervision target downsampling：开启。

复现其他模型时，最公平的做法是复用同一 nnU-Net dataloader/augmentation；如果官方 baseline 训练框架不兼容，也至少要保持 patch、spacing、normalization、foreground oversampling、mirror/rotation/intensity augmentation 的范围一致，并在配置表中说明差异。

## 6. 验证与指标

推荐命令：

```bash
FOLD=0 sbatch revision_bxmnet_experiments/BXM-13_brats2024_full/scripts/submit_validate_fold.sh
```

对应输出：

```bash
revision_bxmnet_experiments/BXM-13_brats2024_full/results/metrics/fold0_metrics.csv
```

评价协议：

- 默认 `RUN_INFERENCE=1`，先运行 nnU-Net validation 导出预测。
- 使用 `checkpoint_best.pth` 路径时要在所有 baseline 中保持一致；若改成 final checkpoint，所有方法同步。
- 保存 case-level CSV，不只保存均值。
- 指标至少包含 WT/TC/ET Dice、AVG Dice、WT/TC/ET HD95、AVG HD95。
- BraTS2024 标签区域按当前验证脚本逻辑：若标签为 `{1,2,3}`，则 `WT={1,2,3}`、`TC={1,3}`、`ET={3}`。
- 统计表建议报告 `mean ± std`，主对比进一步基于 paired per-case metrics 做 Wilcoxon/FDR。

## 7. Baseline 对齐优先级

必须完全一致：

- dataset、fold/split、preprocessing、spacing、patch size、label mapping。
- train/val/test case 列表。
- validation checkpoint 选择规则。
- Dice/HD95 计算脚本与 region 定义。
- case-level CSV 和 prediction masks 保存。

建议一致：

- epochs 或总 iterations。
- batch size 或等效 gradient accumulation。
- optimizer/scheduler。若 baseline 官方训练强依赖自己的 optimizer，可保留官方设置，但要在表中说明。
- augmentation 范围。
- mixed precision、workers、随机种子。

可以保留各模型官方默认：

- 模型内部结构、通道数、window size、transformer/mamba depth。
- 官方建议的 architecture-specific regularization。
- 官方 inference trick，但必须明确是否使用 TTA/sliding-window overlap/probability export。

## 8. 每个 baseline 建议记录的最小配置表

| 字段 | 示例 |
|---|---|
| method | SegMamba / SwinUNETR / nnFormer |
| dataset | `Dataset006_BraTS2024_full` |
| fold | `0` |
| split source | `nnUNet_preprocessed/.../splits_final.json` |
| preprocessing source | `nnUNetPlans`, `3d_fullres` |
| patch size | `[96, 96, 96]` |
| batch size | `1` |
| epochs / iterations | `500 x 250` |
| optimizer | `AdamW(lr=1e-4, wd=1e-4)` or official |
| scheduler | warmup10 + hold20 + cosine eta_min1e-6 or official |
| augmentation | nnU-Net v2 same as BXMNet / official with noted differences |
| checkpoint for evaluation | best |
| metrics script | same Dice/HD95 script |
| outputs | checkpoint, logs, prediction masks, per-case CSV, summary CSV, config snapshot |

## 9. 一句话执行原则

主表复现时，优先保证“同数据、同 split、同预处理、同 patch、同评价脚本、同 checkpoint 选择”。模型专属 optimizer 或训练技巧如果不能统一，不要硬改到失真；要单独记录，让论文里能解释公平性边界。
