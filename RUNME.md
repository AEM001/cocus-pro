# CV 分割管道运行说明

## 概览

这是一个三阶段的语义引导分割管道：
1. **区域提示生成**：将图像划分为编号网格，生成提示图与元数据
2. **初始边界框构建**：基于 LLM 输出的行/列 IDs 生成 prior 掩码与 SAM 框
3. **语义雕刻精修**：使用 Alpha-CLIP 语义评分 + SAM 交互迭代优化掩码


## 运行命令

### 一键全流程
```bash
conda run -n camo-vlm python scripts/main.py --name {name} --text "{instance_text}" --visualize
```

### 分阶段运行

#### 阶段 1：区域提示生成
```bash
conda run -n camo-vlm python scripts/main.py --stage region_prompts --name {name}
```
或直接调用子脚本：
```bash
conda run -n camo-vlm python auxiliary/scripts/make_region_prompts.py --name {name} --outdir auxiliary/out
```

#### 阶段 2：构建初始边界框
```bash
conda run -n camo-vlm python scripts/main.py --stage build_prior --name {name}
```
或直接调用子脚本：
```bash
conda run -n camo-vlm python auxiliary/scripts/build_prior_and_boxes.py --name {name} --out auxiliary/box_out
```

#### 阶段 3：语义雕刻（仅需输入名称）
```bash
conda run -n camo-vlm python -m cog_sculpt.cli --name {name}
```

## 参数说明

### 语义雕刻阶段关键参数
- `--grid 3x3`：初始网格大小（默认3x3）
- `--k 3`：迭代轮数（默认3）
- `--max-points 12`：每轮最大点数（默认12）
- `--use-prior`：使用 prior 掩码作为初始掩码（推荐）
- `--use-boxes`：优先使用 boxes.json 确定 ROI（推荐）
- `--debug`：保存每轮中间结果到 `iter_XX_debug.json`
- `--text "目标文本"`：手动指定目标文本（可选，未指定时自动从 llm_out 读取）

### 边界语义精修策略（已简化）
当前采用**仅沿边界的语义精修**：
- **强制约束**：SAM 分割只能在 prior 掩码约束的框内进行
- **边界采样**：在当前掩码边界均匀采样若干点
- **向内/外探索**：为每个边界点向内和向外生成小圆盘区域
- **Alpha-CLIP 评分**：对圆盘区域进行语义评分
- **点选择**：外侧高分作为正点（扩张），内侧低分作为负点（收缩）
- **迭代更新**：用 SAM 点提示更新掩码，并强制与 prior 求交

## 输出结果

### 最终输出
- **掩码**：`pipeline_output/{name}/final_mask.png`
- **叠加可视化**：`pipeline_output/{name}/final_overlay.png`

### 调试输出（--debug 模式）
- **每轮掩码**：`pipeline_output/{name}/iter_XX_mask.png`
- **每轮叠加**：`pipeline_output/{name}/iter_XX_overlay.png`
- **每轮调试数据**：`pipeline_output/{name}/iter_XX_debug.json`

## 示例：处理名为 "f" 的图像
```bash
# 准备文件
# auxiliary/images/f.png
# auxiliary/llm_out/f_output.json
# auxiliary/box_out/f/f_sam_boxes.json
# auxiliary/box_out/f/f_prior_mask.png

# 运行语义雕刻（仅需输入名称）
conda run -n camo-vlm python -m cog_sculpt.cli --name f
```

## 故障排除

1. **权重文件缺失**：确保 `checkpoints/` 和 `models/` 目录下有对应的权重文件
2. **输入文件缺失**：检查 `auxiliary/images/` 和 `auxiliary/llm_out/` 下是否有对应名称的文件
3. **环境问题**：确保在 `camo-vlm` conda 环境中运行所有命令
4. **内存不足**：可以尝试使用 CPU 版本的 torch 或减小 `--max-points` 参数

## 简化的边界语义精修策略

系统采用**仅沿边界的语义精修**，完全移除网格路径：
- **约束框架**：SAM 分割始终约束在 prior 掩码区域内
- **边界采样**：在当前掩码的边界处采样 24 个候选点
- **双向探索**：为每个边界点向内和向外探索小圆盘区域（半径 8 像素）
- **语义评分**：用 Alpha-CLIP 对圆盘区域进行多模板语义评分
- **智能选点**：外侧高分作为正点（扩张），内侧低分作为负点（收缩）
- **迭代更新**：用 SAM 点提示更新掩码，并强制与 prior 求交限制
- **自动早停**：当 IoU 变化 < 0.5% 时停止迭代
