# 语义引导的迭代雕刻分割（Alpha‑CLIP × SAM）

基于 guide.md 的“ROI → 子格 → 语义打分 → 正/负点 → SAM 迭代”流程，实现 Alpha 通道软先验的语义评分与多轮交互式分割（k≈3）。

## 核心特性

- 4 通道 Alpha‑CLIP：以掩码为 Alpha 通道进行子区域语义评分（多模板聚合）。
- 语义引导点：自适应阈值选择正/负点（质心/边界外点）。
- SAM 迭代收敛：每轮平滑掩码，逐步细化边界与抑制外溢。
- 模块化骨架：可替换打分器、SAM 实现与点选择策略。

## 代码结构

- `alpha_clip_rw/`
  - `alpha_clip.py`：加载 CLIP/Alpha‑CLIP，图像与 Alpha 变换。
  - `inference.py`：`AlphaCLIPInference` 推理封装，`score_region_with_templates()` 多模板聚合。
  - `model.py`：包含视觉干预（额外 Alpha 分支）的模型定义。
- `cog_sculpt/`
  - `core.py`：主算法管线与工具（ROI/网格/打分/点选择/平滑/IoU/SAM 接口）。入口函数：`sculpting_pipeline()`。
  - `cli.py`：命令行入口（读取路径、装配 scorer 与 SAM、保存中间与最终结果）。
- `sam_integration.py`：
  - `SAMWrapper`/`SAM2Wrapper`：统一封装 `set_image / predict_with_box / predict_with_points`。
  - `SemanticGuidedSAM`：简化版“网格打分 + 点交互”的一体化类。
- `auxiliary/`
  - `scripts/make_region_prompts.py`：图片 9×9（可配）行/列提示图与 `meta.json`。
  - `scripts/build_prior_and_boxes.py`：由 LLM 输出 ids 生成 prior 掩码与 `sam_boxes.json`。
  - `llm_out/ | out/ | box_out/ | images/`：示例输入与中间产物。
- `scripts/main.py`：流水线脚本（便于串起各阶段；本地直接用模块命令更直观，见下）。

## 流程对照（guide.md → 实现）

- Step 0 输入：图像、ROI（由 ids/box 推回）、初始掩码 M0、文本 T。
  - 已实现：`cog_sculpt/cli.py` 通过 `boxes.json` 或 prior 掩码推 ROI，M0 可直接用 prior。
- Step 1 网格化：ROI 内规则分块（初始 3×3）。
  - 已实现：`partition_grid()`；提供 `subdivide_cell()`（递归细分未在主循环启用）。
- Step 2 Alpha 软先验评分：RGBA 4 通道，支持多模板平均。
  - 已实现：`AlphaCLIPInference.score_region_with_templates()`；`AlphaCLIPScorer` 调用。
- Step 3 阈值与选点：μ±0.5σ，自适应正/负阈值；质心正点 + 边界外负点。
  - 已实现：`threshold_cells()`、`select_points_from_cells()`（含边界带搜索）。
- Step 4 SAM 交互与早停：点提示更新掩码 + 平滑；判断收敛后停止。
  - 已实现：`SamWrapper.predict_points()` + `smooth_mask()`。
  - 早停注意：当前用 `iou(M_new, M)` 判断，建议阈值逻辑参见“已知差异”。
- Step 5 迭代（k≈3）：默认 3 轮，可早停。
  - 已实现：`Config.k_iters=3`。

## 环境与依赖

- 必需（基础运行与工具脚本）：`numpy`, `Pillow`, `PyYAML`, `opencv-python`
- Alpha‑CLIP（可选）：`torch` (>=1.7.1), `torchvision`, `tqdm`, `packaging`, `loralib`
- SAM 推理（可选但强烈建议）：`segment-anything`（或 `sam2`），以及对应模型权重（见其官方仓库）

提示：当前仓库未提供 `requirements.txt`。如需使用 `setup.py` 的 console_scripts，请先自行安装依赖；或直接使用下方的“模块命令行”方式运行。

## 运行示例（推荐的模块命令）

以内置示例 `auxiliary/images/q.png` 与现成 ids/meta 为例：

1) 由 ids 生成 prior 与 SAM 框（也可直接使用仓库里已有的 `auxiliary/box_out/q/*`）

```bash
python auxiliary/scripts/build_prior_and_boxes.py \
  --image auxiliary/images/q.png \
  --meta auxiliary/out/q/q_meta.json \
  --pred auxiliary/llm_out/q_output.json \
  --out auxiliary/box_out
```

会得到：
- `auxiliary/box_out/q/q_prior_mask.png`
- `auxiliary/box_out/q/q_sam_boxes.json`

2) 语义雕刻（需要已安装 SAM；Alpha‑CLIP 可选，未安装会回退到简单打分器）

```bash
python -m cog_sculpt.cli \
  --text "scorpionfish" \
  --image auxiliary/images/q.png \
  --boxes auxiliary/box_out/q/q_sam_boxes.json \
  --prior-mask auxiliary/box_out/q/q_prior_mask.png \
  --name q \
  --out-root pipeline_output \
  --use-prior --use-boxes --debug
```

输出：
- `pipeline_output/q/final_mask.png`
- `pipeline_output/q/final_overlay.png`
- 如启用 `--debug`，还会生成每轮点与分数的 `iter_**_debug.json`。

3) 仅用 Python API（一体化类）

```python
from alpha_clip_rw import AlphaCLIPInference
from sam_integration import SemanticGuidedSAM, create_sam_wrapper

alpha_clip = AlphaCLIPInference("ViT-L/14@336px")  # 可选
sam = create_sam_wrapper(model_type="sam", checkpoint_path="models/sam_vit_h_4b8939.pth")
sgs = SemanticGuidedSAM(sam, alpha_clip)

mask = sgs.segment_with_semantic_points(
    image, bbox=[x1, y1, x2, y2], instance_text="scorpionfish", iterations=3
)
```

## 已知差异与建议（相对 guide.md）

- `cog_sculpt/core.py:487` 早停条件目前为 `if iou(M_new, M) < cfg.iou_eps: break`，与“收敛即 IoU 较高”直觉相反。
  - 建议改为高 IoU 提前停止，例如：`if iou(M_new, M) > (1 - cfg.iou_eps): break` 或直接用固定阈值如 `> 0.99`。
- 不确定格递归细分：提供了 `subdivide_cell()`，但主循环暂未启用递归（Step 3 的可选增强）。如需更稳结果，可对 `unc` 子格做最多 1–2 层 2×2 细分后再阈分与落点。
- `scripts/main.py` 作为统一入口依赖内部 argparse，当前与 `cog_sculpt/cli.py` 的参数对接有限。建议优先使用上文“模块命令行”示例，或后续将 `scripts/main.py` 调整为直接传参调用而非二次解析。
- 文档链接与配置：原 README 中的 `docs/USAGE.md`、`docs/API.md`、`auxiliary/docs/config.yaml` 不存在。本 README 已改为直接给出可运行命令与路径。

## 参考与致谢

- Segment Anything (SAM)
- Alpha‑CLIP / CLIP

如果你希望，我也可以补全上面提到的早停条件与递归细分的小改动，并补一份可选的 `requirements.txt` 以便一键安装。
