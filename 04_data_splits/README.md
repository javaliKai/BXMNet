# BXMNet 数据划分与病例 ID

本目录保存 BXMNet 实验使用的固定病例清单，不包含影像数据本身。

## 固定 80:20 划分

- `brats2019_80_20_268_67/`：268 train / 67 val
- `brats2020_80_20_295_74/`：295 train / 74 val
- `brats2024_80_20_1080_270/`：1080 train / 270 val

每个目录中的 `splits_final.json` 可作为 nnU-Net split 文件，`train_ids.txt`、
`val_ids.txt` 和 `cases.csv` 是便于检查与复现的病例清单。

## BraTS2020 五折

`brats2020_train295_5fold/` 是固定 295 例训练池上的五折划分。每折为：

- 训练：236 例
- 验证：59 例

该划分继承已有 294 例 Dataset007 五折顺序，并将缺失的官方病例
`BraTS20_Training_355` 放入 fold 4 的验证集，使五折大小一致。

当前本地 BraTS2020 原始目录没有该病例的影像和标签，恢复数据后再运行训练。
