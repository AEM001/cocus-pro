# 🚀 批量处理使用指南

## 📋 概述

批量处理系统可以自动处理 `auxiliary/images/` 目录下的所有图像文件，支持完整的目标检测到SAM精修流程。

### ✨ 主要特性

- **🔄 自动化流程**：网格生成 → API目标检测 → SAM精修 → 输出清理
- **🎯 智能目标映射**：内置常见图像的目标检测查询
- **⚡ 并行处理**：支持多线程并行加速
- **🛡️ 错误重试**：失败自动重试机制
- **📊 详细报告**：处理结果统计和错误分析
- **🧹 输出清理**：只保留最终掩码和instance信息

## 🏗️ 输出结构

### 标准模式输出
```
outputs/clean_sculpt/[image_name]/
├── final_mask.png          # 最终分割掩码
├── instance_info.txt       # 实例信息文件
├── final_visualization.png # 可视化结果
└── round*_step*_*.png      # 中间步骤文件
```

### 清理模式输出 (`--clean-output`)
```
outputs/clean_sculpt/[image_name]/
├── final_mask.png     # 最终分割掩码
└── instance_info.txt  # 实例信息文件
```

### instance_info.txt 内容示例
```
Sample: f
Instance: scorpionfish
Description: The scorpionfish is camouflaged against the sandy seabed...
Final mask pixels: 62645
ROI: (427.0, 320.0, 995.0, 639.0)
Processing rounds: 4
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活环境并设置API密钥
conda activate camo-vlm
export DASHSCOPE_API_KEY="your_api_key_here"
```

### 2. 基本使用
```bash
# 处理所有图像（使用API模式，清理输出）
python batch_process.py --use-api --clean-output

# 处理指定图像
python batch_process.py --use-api --include f dog cat

# 排除某些图像
python batch_process.py --use-api --exclude person person2
```

### 3. 并行处理
```bash
# 使用4个并行进程
python batch_process.py --use-api --parallel 4 --clean-output
```

## 📋 完整命令参考

### 基本参数
```bash
--image-dir DIR           # 图像目录 (默认: auxiliary/images)
--include NAME [NAME...]  # 只处理指定图像 (不含扩展名)
--exclude NAME [NAME...]  # 排除指定图像 (不含扩展名)
--target "QUERY"         # 统一的目标检测查询
```

### API参数
```bash
--api-key KEY           # API密钥 (或设置环境变量)
--api-model MODEL       # API模型 (默认: qwen-vl-plus-latest)
--use-api              # 使用API模式
--use-openai-api       # 使用OpenAI兼容模式
--high-resolution      # 启用高分辨率模式
```

### 处理参数
```bash
--grid-size SIZE       # 网格大小 (默认: 9)
--rounds NUM           # SAM精修轮数 (默认: 4)
--ratio FLOAT          # 象限比例 (默认: 0.8)
--vlm-max-side SIZE    # VLM输入最大边长 (默认: 720)
--output-format FORMAT # 输出格式 png/jpg (默认: png)
--clean-output         # 只保留最终掩码和instance信息
```

### 批处理参数
```bash
--parallel NUM         # 并行处理数 (默认: 1)
--retry-count NUM      # 失败重试次数 (默认: 2)
--report-dir DIR       # 报告保存目录 (默认: batch_reports)
```

### 跳过步骤选项
```bash
--skip-grid           # 跳过网格生成
--skip-detection      # 跳过目标检测  
--skip-build          # 跳过SAM输入生成
--skip-sculpt         # 跳过SAM精修
```

## 💡 使用场景示例

### 场景1：生产环境批量处理
```bash
# 推荐配置：API模式，清理输出，并行处理
python batch_process.py \
    --use-api \
    --clean-output \
    --parallel 3 \
    --api-model qwen-vl-plus-latest \
    --rounds 3 \
    --output-format png
```

### 场景2：测试单个图像
```bash
# 处理单个图像，保留所有中间文件用于调试
python batch_process.py \
    --use-api \
    --include f \
    --rounds 2 \
    --ratio 0.8
```

### 场景3：大批量处理
```bash
# 高并行，高效处理，成本优化
python batch_process.py \
    --use-api \
    --clean-output \
    --parallel 5 \
    --api-model qwen-vl-plus-latest \
    --rounds 2 \
    --ratio 0.6 \
    --vlm-max-side 512
```

### 场景4：自定义目标检测
```bash
# 使用统一的目标查询，覆盖默认映射
python batch_process.py \
    --use-api \
    --target "find the hidden animal in the image" \
    --include cat dog person
```

### 场景5：分步处理
```bash
# 只执行目标检测，不做SAM精修
python batch_process.py \
    --use-api \
    --skip-sculpt

# 只执行SAM精修 (假设已有检测结果)
python batch_process.py \
    --use-api \
    --skip-grid \
    --skip-detection \
    --skip-build
```

## 📊 报告和监控

### 实时进度
```
============================================================
🚀 批量图像处理系统
============================================================
API密钥: sk-a2047...
API模型: qwen-vl-plus-latest
待处理图像: 5
  - f
  - dog
  - cat
  - q
  - person2
处理模式: API
并行数: 3
清理输出: 是

[1/5] 处理 f...
============================================================
🖼️  处理图像: f
   路径: auxiliary/images/f.png
============================================================
[10:45:23] 生成网格标注图
[10:45:25] API目标检测
[10:45:28] 生成SAM输入
[10:45:30] SAM精修
✅ f 处理完成! 用时: 45.2秒
```

### 批处理报告
系统会自动生成两种报告：

#### 1. JSON详细报告 (`batch_report_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "2025-09-17T02:59:07",
  "summary": {
    "total_images": 5,
    "successful": 4,
    "failed": 1,
    "success_rate": 80.0,
    "total_processing_time": 234.5,
    "average_processing_time": 46.9
  },
  "results": [...]
}
```

#### 2. 文本摘要报告 (`batch_summary_YYYYMMDD_HHMMSS.txt`)
```
============================================================
批量处理报告
============================================================
处理时间: 2025-09-17 10:59:07
总图像数: 5
成功: 4
失败: 1
成功率: 80.0%
总处理时间: 234.5秒
平均处理时间: 46.9秒/图像

成功处理的图像:
----------------------------------------
✅ f (45.2s)
   输出文件: final_mask.png, instance_info.txt
✅ dog (52.1s)
   输出文件: final_mask.png, instance_info.txt
...

失败的图像:
----------------------------------------
❌ person2: API调用失败: Connection timeout
```

## ⚙️ 默认目标映射

系统内置了常见图像的目标检测查询：

```python
DEFAULT_TARGETS = {
    'f': 'find the camouflaged scorpionfish',
    'dog': 'find the camouflaged dog', 
    'cat': 'find the camouflaged cat',
    'q': 'find the camouflaged animal',
    'person': 'find the camouflaged person',
    'person2': 'find the camouflaged person'
}
```

如果图像名不在映射中，会使用通用查询：`find the camouflaged object in {name}`

## 🛡️ 错误处理和重试

### 自动重试机制
- API调用失败自动重试 (默认2次)
- 网络超时自动重试
- 临时错误自动重试

### 常见错误处理
1. **API密钥错误**：检查环境变量设置
2. **网络超时**：增加重试次数 `--retry-count 3`
3. **内存不足**：降低并行数 `--parallel 1`
4. **磁盘空间不足**：使用 `--clean-output` 清理中间文件

## 📈 性能优化建议

### 1. API成本优化
```bash
# 使用plus模型，降低图像尺寸，减少轮数
--api-model qwen-vl-plus-latest \
--vlm-max-side 512 \
--rounds 2
```

### 2. 处理速度优化
```bash
# 适度并行，跳过不必要步骤
--parallel 3 \
--clean-output \
--skip-grid  # 如果已有网格文件
```

### 3. 存储空间优化
```bash
# 清理输出，使用jpg格式
--clean-output \
--output-format jpg
```

## 🔧 故障排除

### 问题1: 图像文件找不到
```
[ERROR] 在 auxiliary/images 中未找到支持的图像文件
```
**解决方案**：确保图像文件在正确目录，格式受支持

### 问题2: API调用失败
```
[ERROR] API调用失败: Connection timeout
```
**解决方案**：检查网络连接，增加重试次数，检查API密钥

### 问题3: 内存不足
```
CUDA out of memory
```
**解决方案**：降低并行数，减少图像尺寸，使用CPU模式

### 问题4: 权限错误
```
Permission denied: outputs/clean_sculpt/
```
**解决方案**：检查目录权限，创建输出目录

## 📞 技术支持

如遇问题，请检查：
1. **环境配置**：conda环境激活，API密钥设置
2. **文件路径**：图像文件存在，路径正确
3. **网络连接**：API服务可访问
4. **系统资源**：内存和磁盘空间充足
5. **批处理报告**：查看详细错误信息

---

**🎉 现在你可以高效批量处理图像了！开始体验自动化的分割系统吧！**