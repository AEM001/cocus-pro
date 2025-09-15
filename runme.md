# SAM优先VLM雕刻工作流中文使用指南

## 快速开始

- 激活环境并设置分词器行为

```bash
conda activate camo-vlm
export TOKENIZERS_PARALLELISM=false
```

- 使用本地Qwen2.5-VL-AWQ和自动SAM运行

```bash
python scripts/run_sculpt_simple.py <名称> [--轮数 N]
```

## 使用示例

```bash
# 使用Qwen（默认），自动检测models/下的SAM检查点
python scripts/run_sculpt_simple.py f --rounds 2

# 使用Mock VLM（无需GPU）
python scripts/run_sculpt_simple.py f --rounds 2 --model mock

# 显式指定SAM检查点和设备
python scripts/run_sculpt_simple.py f \
  --sam_checkpoint models/sam_vit_h_4b8939.pth \
  --sam_type vit_h \
  --sam_device cuda \
  --rounds 2

# 使用自定义Qwen模型目录
python scripts/run_sculpt_simple.py f --model qwen --model_dir models/Qwen2.5-VL-7B-Instruct-AWQ
```

## 标准数据布局

- 输入图像：`auxiliary/images/{名称}.png`
- ROI + 先验：`auxiliary/box_out/{名称}/{名称}_sam_boxes.json` 和 `{名称}_prior_mask.png`
- 实例说明：`auxiliary/llm_out/{名称}_output.json`（字段：instance）
- 输出：`outputs/sculpt/{名称}/`

## 脚本执行流程

1) 自动加载图像、ROI、先验掩码和实例
2) 从ROI框生成初始SAM掩码（严格裁剪到ROI内）
3) 执行N轮VLM驱动雕刻
4) 保存每轮控制点、peval JSON、分数JSON、掩码和最终输出

## 所有可配置参数

### scripts/run_sculpt_simple.py

- **name**（位置参数）
  - 目标样本名称，匹配auxiliary/下的文件

- **--rounds int**（默认：2）
  - 雕刻轮数

- **--model [mock|qwen]**（默认：qwen）
  - 选择VLM后端

- **--model_dir str**（默认：models/Qwen2.5-VL-7B-Instruct-AWQ）
  - 本地Qwen2.5-VL模型目录路径

- **--sam_checkpoint str**（可选）
  - SAM检查点路径(.pth)。如省略，脚本自动扫描models/下类似sam_*vit_[hlb]*.pth的检查点

- **--sam_type [vit_h|vit_l|vit_b]**（默认：vit_h）
  - 与检查点匹配的SAM模型类型

- **--sam_device [cuda|cpu]**（可选）
  - SAM设备。如省略，由torch推断

- **--K_pos int**（默认：6）
  - 每轮正样本点数

- **--K_neg int**（默认：6）
  - 每轮负样本点数

## 环境变量

- **TOKENIZERS_PARALLELISM=false**
  - 防止分词器生成并行工作进程导致警告

## 注意事项

- SAM初始掩码严格裁剪到ROI框内，确保初始化包含在内
- 候选采样优先边界点，密度根据边界长度自适应
- Patch尺度较小且基于目标尺寸（掩码bbox），带轻微上下文以捕获边界两侧