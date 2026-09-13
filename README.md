# BXMNet 代码与实验协议整理包

这是从 `dcr_resdec` 仓库整理出的 BXMNet 方法代码与实验协议副本，原仓库文件保持不变。

## 目录结构

```text
BXMNet/
├── 01_core_model/              # BXMNet、DCR encoder、Mamba neck、MAFCM decoder
├── 02_nnunet_integration/      # nnU-Net trainers、训练/验证入口和运行脚本
├── 03_configs_and_training/    # 模型配置、plans、训练参数和实际配置快照
└── 04_data_splits/             # 固定 80:20 划分与 BraTS2020 五折病例 ID
```

## 关键文件

- 核心模型入口：`01_core_model/bxmnet.py`
- 基础 nnU-Net trainer：`02_nnunet_integration/trainers/nnUNetTrainerBXMNet.py`
- 修订版 trainer：`02_nnunet_integration/trainers/nnUNetTrainerBXMNetRevision.py`
- 训练参数：`03_configs_and_training/configs/training_parameters.md`
- 固定划分说明：`04_data_splits/README.md`

## 使用说明

这是一个按功能归档的整理副本。核心代码内部仍使用原项目的
`nnunetv2.training.nnUNetTrainer.bxmnet...` import 路径，因此实际运行时仍需要完整的
nnU-Net 和 `dynamic-network-architectures` 依赖；本目录不复制整个 nnU-Net 框架。

BraTS2020 的逻辑固定划分为 295/74，但当前原始数据缺少
`BraTS20_Training_355`；恢复病例文件后才能直接使用该清单训练。
