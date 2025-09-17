# SAM Sculpt 快速使用指南

## 核心功能
SAM Sculpt 是一个结合 SAM（Segment Anything Model）和 Qwen2.5-VL 的智能图像分割优化工具，通过多轮对话实现精确分割。

## 单图处理

### 基础用法
```bash
python clean_sam_sculpt.py --name sample_name
```

### 关键参数

#### 1. 精修控制参数
- `--rounds N` - SAM精修轮数 (默认: 1)
  ```bash
  python clean_sam_sculpt.py --name f --rounds 2
  ```

- `--ratio 0.6` - 象限比例，控制分析区域大小 (默认: 0.6)
  ```bash  
  python clean_sam_sculpt.py --name f --ratio 0.8
  ```

#### 2. VLM模型配置
- `--vlm-max-side N` - VLM输入图像最大边长 (默认: 720)
  ```bash
  python clean_sam_sculpt.py --name f --vlm-max-side 1024
  ```

- `--qwen_dir PATH` - 本地Qwen模型路径
  ```bash
  python clean_sam_sculpt.py --name f --qwen_dir /path/to/qwen/model
  ```

#### 3. API模式 (推荐)
```bash
# 使用通义千问API
export DASHSCOPE_API_KEY="your-api-key"
python clean_sam_sculpt.py --name f --use-api

# 使用OpenAI兼容API + 高分辨率
python clean_sam_sculpt.py --name f --use-openai-api --high-resolution
```

#### 4. 输出格式控制
- `--output-format png/jpg` - 输出掩码格式 (默认: png)
- `--clean-output` - 只保留最终掩码和实例信息
  ```bash
  python clean_sam_sculpt.py --name f --clean-output --output-format jpg
  ```

### 完整示例
```bash
# 推荐配置：使用API + 1轮精修 + 清理输出
python clean_sam_sculpt.py --name dog \\
    --use-api \\
    --rounds 1 \\
    --ratio 0.6 \\
    --vlm-max-side 1024 \\
    --clean-output \\
    --output-format png
```

## 批处理

### 基础批处理
```bash
python batch_process.py --samples f,dog,cat
```

### 带参数批处理
```bash
python batch_process.py \\
    --samples f,dog,cat \\
    --rounds 1 \\
    --use-api \\
    --clean-output \\
    --parallel 2
```

### 自动化批处理（扫描所有可用样本）
```bash
python batch_process.py --auto --use-api --rounds 1
```

## 输入文件要求

### 必需文件
1. **原始图片**: `auxiliary/images/{name}.png`
2. **ROI框信息**: `auxiliary/box_out/{name}/{name}_sam_boxes.json`

### 可选文件  
3. **语义信息**: `auxiliary/llm_out/{name}_output.json` (包含实例名称和推理描述)

### ROI框文件格式示例
```json
{
    "x0": 100.5,
    "y0": 150.2,
    "x1": 400.8,
    "y1": 350.9
}
```

## 输出结果

### 输出目录结构
```
outputs/clean_sculpt/{sample_name}/
├── round0_step1_initial_mask.png          # 初始SAM掩码
├── round1_step2_anchors.png                # 第1轮锚点图
├── round1_step3_selected_crop.png          # VLM选择的区域
├── round1_step4_refined_mask.png           # 第1轮精修结果
├── ...                                     # 各轮中间结果
├── final_mask.png                          # 最终掩码 
└── instance_info.json                      # 实例信息
```

### 实例信息文件示例
```json
{
    "instance_name": "dog", 
    "semantic_reason": "目标是一只棕色的拉布拉多犬...",
    "processing_rounds": 3,
    "final_bbox": [100.5, 150.2, 400.8, 350.9]
}
```

## 性能优化建议

### 1. API vs 本地模型
- **API模式**: 更快，结果更好，推荐用于生产
- **本地模式**: 适合离线环境，需要较好的GPU

### 2. 参数调优
- **轮数**: 1轮即可得到良好结果，更多轮数收益递减
- **比例**: 0.6-0.8 适合大多数场景
- **分辨率**: API模式可用高分辨率，本地建议720-1024

### 3. 批处理优化
```bash
# 并行处理 + API + 清理输出
python batch_process.py --auto \\
    --use-api \\
    --rounds 1 \\
    --parallel 4 \\
    --clean-output
```

## 故障排查

### 1. API连接问题
- 检查网络连接和API密钥
- 尝试添加重试机制: `--max-retries 3`

### 2. 内存不足
- 降低 `--vlm-max-side` 参数
- 减少并行数量 `--parallel`

### 3. 分割效果不佳
- 增加精修轮数 `--rounds` (默认1轮已足够)
- 调整象限比例 `--ratio`
- 检查输入ROI框是否准确

## 环境变量

```bash
# API密钥
export DASHSCOPE_API_KEY="your-dashscope-key"

# 本地模型路径 (可选)
export QWEN_VL_MODEL_DIR="/path/to/qwen/model"

# CUDA设备 (可选)
export CUDA_VISIBLE_DEVICES="0"
```