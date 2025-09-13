# Auxiliary Files - 辅助文件

本文件夹包含项目的辅助文件和非核心内容，已按功能分类整理。

## 📁 目录结构

### scripts/ - 辅助脚本
- `make_region_prompts.py` - 图片切分与标号实现，输出到 data/out
- `build_prior_and_boxes.py` - 初步边框和掩码划分，输出到 data/box_out  
- `sam.py` - SAM 相关工具函数

### docs/ - 文档配置
- `config.yaml` - 项目配置文件
- `prompt.md` - VLM 定位使用的提示词
- `guide.md` - 项目指南文档
- `code_guide.md` - 详细的代码实现指南
- `README_SCULPT.md` - Cog-Sculpt 原型说明

### data/ - 数据文件
- `images/` - 初始测试图片
- `llm_out/` - VLM 输出的 JSON 信息  
- `out/` - make_region_prompts.py 的图片切分输出结果
- `box_out/` - build_prior_and_boxes.py 的边框和掩码划分输出

## 🔧 使用说明

这些文件支持主要工作流程中的数据预处理、配置管理和结果存储等功能。如需使用某个脚本，可以从项目根目录运行：

```bash
python auxiliary/scripts/make_region_prompts.py
python auxiliary/scripts/build_prior_and_boxes.py
```

配置文件和文档可在需要时参考或修改。