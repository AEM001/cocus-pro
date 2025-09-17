# 🚀 阿里云百炼 Qwen VL API 集成指南

本文档描述如何使用阿里云百炼平台的 Qwen VL API 替换原有的本地模型调用。

## 📋 功能概述

项目中总共有 **3个大模型使用点**，已全部支持API调用：

1. **目标检测** - 网格化目标定位 (auxiliary 流程)
2. **锚点选择** - 选择需要精修的锚点 (clean_sam_sculpt.py) 
3. **象限编辑** - 生成正负点编辑指令 (clean_sam_sculpt.py)

## 🛠️ 环境准备

### 1. 安装依赖

```bash
# 自动安装所有必要依赖
python install_api_deps.py

# 或手动安装
pip install dashscope openai>=1.0.0 pillow numpy requests
```

### 2. 配置API密钥

获取阿里云百炼API密钥后，设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

或者在代码中直接传递：
```bash
python script.py --api-key "your_api_key_here"
```

## 🎯 使用方法

### 方式1: 完整流程自动化

```bash
# 运行完整的API流程（推荐）
python run_api_pipeline.py --name f --target "find the camouflaged scorpionfish"

# 只执行目标检测，不运行SAM精修
python run_api_pipeline.py --name f --only-detection

# 使用高分辨率模式
python run_api_pipeline.py --name f --high-resolution

# 使用OpenAI兼容模式
python run_api_pipeline.py --name f --use-openai-api

# 自定义模型和参数
python run_api_pipeline.py --name f \
    --model qwen-vl-max-latest \
    --rounds 6 \
    --ratio 0.5 \
    --high-resolution
```

### 方式2: 分步执行

#### 步骤1: 生成网格标注图
```bash
cd auxiliary/scripts
python make_region_prompts.py --name f --rows 9 --cols 9
```

#### 步骤2: API目标检测
```bash
cd auxiliary/scripts
python detect_target_api.py --name f --target "find the camouflaged scorpionfish"
```

#### 步骤3: 生成SAM输入
```bash
cd auxiliary/scripts  
python build_prior_and_boxes.py --name f
```

#### 步骤4: SAM精修 (使用API)
```bash
python clean_sam_sculpt.py --name f --use-api --rounds 4 --ratio 0.8
```

### 方式3: 仅替换SAM精修部分

如果你已有 `auxiliary/llm_out/f_output.json` 和 `auxiliary/box_out/f/f_sam_boxes.json`，可以直接使用API进行SAM精修：

```bash
python clean_sam_sculpt.py --name f --use-api --api-model qwen-vl-plus-latest
```

## ⚙️ API配置参数

### 模型选择
- `qwen-vl-plus-latest` - 推荐，性价比高
- `qwen-vl-max-latest` - 最强性能，成本较高

### API模式选择
- **DashScope SDK** (默认，推荐) - 支持本地文件直传，更稳定
- **OpenAI兼容模式** - 使用Base64编码，兼容性好

### 高分辨率模式
- `--high-resolution` - 提升图像细节理解，但增加token消耗

## 📊 API调用量估算

以单个样本为例：
- **目标检测**: 1次调用 (网格定位)
- **锚点选择**: 4轮 × 1次 = 4次调用  
- **象限编辑**: 4轮 × 1.5锚点 = 6次调用

**总计约 11次API调用/样本**

## 💰 成本优化建议

1. **模型选择**: 优先使用 `qwen-vl-plus-latest`
2. **图像尺寸**: 设置合适的 `--vlm-max-side` (默认720px)
3. **轮数控制**: 根据需要调整 `--rounds` 参数
4. **批量处理**: 对多个样本可以并行处理

## 🔧 API参数详解

### 目标检测API (`detect_target_api.py`)
```bash
python detect_target_api.py \
    --name f \                          # 样本名称
    --target "find the scorpionfish" \  # 目标描述  
    --model qwen-vl-plus-latest \       # 模型名称
    --grid-size 9 \                     # 网格大小
    --high-res \                        # 高分辨率模式
    --use-openai                        # 使用OpenAI兼容模式
```

### SAM精修API (`clean_sam_sculpt.py`)
```bash
python clean_sam_sculpt.py \
    --name f \                          # 样本名称
    --use-api \                         # 启用API模式
    --api-model qwen-vl-plus-latest \   # API模型
    --high-resolution \                 # 高分辨率模式
    --use-openai-api \                  # OpenAI兼容模式
    --rounds 4 \                        # 精修轮数
    --ratio 0.8 \                       # 象限比例
    --vlm_max_side 720                  # 图像最大边长
```

## 📝 输出文件说明

### 目标检测输出
- `auxiliary/llm_out/f_output.json` - VLM检测结果
```json
{
  "instance": "camouflaged scorpionfish",
  "ids_line_vertical": [3, 4, 5, 6, 7],
  "ids_line_horizontal": [5, 6, 7, 8],
  "reason": "检测原因说明..."
}
```

### SAM输入文件  
- `auxiliary/box_out/f/f_sam_boxes.json` - SAM边界框
- `auxiliary/box_out/f/f_prior_mask.png` - 初始掩码

### 最终结果
- `outputs/clean_sculpt/f/final_result.png` - 精修后掩码
- `outputs/clean_sculpt/f/final_visualization.png` - 可视化结果

## ⚠️ 注意事项

1. **API密钥安全**: 不要在代码中硬编码API密钥，使用环境变量
2. **网络稳定性**: API调用需要稳定的网络连接
3. **错误重试**: 脚本包含自动重试机制，但建议监控API调用状态
4. **成本控制**: 注意API调用频率和token消耗
5. **文件清理**: API模式会产生临时文件，程序会自动清理

## 🐛 故障排除

### 1. API密钥错误
```
[ERROR] API Key is required. Set DASHSCOPE_API_KEY environment variable
```
**解决**: 检查环境变量设置 `echo $DASHSCOPE_API_KEY`

### 2. 网络连接问题
```
[ERROR] API调用失败: Connection timeout
```
**解决**: 检查网络连接，必要时重试

### 3. 依赖缺失
```
[ERROR] DashScope SDK not installed
```
**解决**: 运行 `python install_api_deps.py` 安装依赖

### 4. JSON解析失败
```
[ERROR] JSON解析失败: Expecting value
```
**解决**: 脚本包含JSON修复机制，通常会自动恢复

## 📈 性能对比

| 模式 | 速度 | 成本 | GPU需求 | 稳定性 |
|------|------|------|---------|--------|
| 本地模型 | 快 | 无 | 7-8GB | 高 |
| API调用 | 中等 | 有 | 无 | 高 |

## 🔄 迁移指南

### 从本地模型迁移到API

1. **安装依赖**: `python install_api_deps.py`
2. **设置密钥**: `export DASHSCOPE_API_KEY="..."`
3. **测试API**: `python run_api_pipeline.py --name f --only-detection`
4. **完整迁移**: 在所有脚本中添加 `--use-api` 参数

### API模式选择建议

- **DashScope SDK** (推荐): 更稳定，支持文件直传
- **OpenAI兼容**: 便于集成其他工具链

## 🚀 快速开始

```bash
# 1. 安装依赖
python install_api_deps.py

# 2. 设置API密钥
export DASHSCOPE_API_KEY="sk-xxx"

# 3. 运行测试
python run_api_pipeline.py --name f --target "find the camouflaged scorpionfish" --only-detection

# 4. 完整流程
python run_api_pipeline.py --name f
```

## 📞 技术支持

如果遇到问题，请检查：
1. API密钥是否正确设置
2. 网络连接是否正常  
3. 依赖包是否正确安装
4. 输入文件是否存在

---

**🎉 现在你可以使用阿里云百炼API来加速你的视觉分割任务了！**