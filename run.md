下面给你一份“可直接运行的命令清单”，按模块和文件划分，并结合本仓库当前实现与默认路径约定。所有命令都以项目根目录为工作目录执行：`/Users/Mac/code/cocus-pro/`

先决条件
- 权重与依赖
  - SAM 权重文件：`models/sam_vit_h_4b8939.pth`
  - Alpha-CLIP 视觉分支权重：`checkpoints/clip_l14_336_grit_20m_4xe.pth`
  - 可选本地 CLIP 主干：`checkpoints/ViT-L-14-336px.pt`
  - 建议设置环境变量（可选，Alpha-CLIP 会自动搜寻）：
    ```bash
    export ALPHA_CLIP_ALPHA_CKPT=checkpoints/clip_l14_336_grit_20m_4xe.pth
    export CLIP_MODEL_PATH=checkpoints/ViT-L-14-336px.pt
    ```
- 数据约定
  - 输入图像：`auxiliary/images/{name}.png`（或 .jpg）
  - 区域提示与 meta：`auxiliary/out/{name}/`
  - LLM 输出（ids）：`auxiliary/llm_out/{name}_output.json`
  - prior 掩码与 boxes：`auxiliary/box_out/{name}/`
  - 最终雕刻输出：`pipeline_output/{name}/`

一、主流水线入口 scripts/main.py
文件：`scripts/main.py`

- 全流程（region_prompts → build_prior → sculpt）
  ```bash
  python -m scripts.main --name q --text "scorpionfish" --visualize --debug
  ```
  说明：
  - 需要已准备 `auxiliary/llm_out/q_output.json`
  - 输出结果在 `pipeline_output/q/`，调试产物在同目录下

- 只跑 区域提示图（生成 `auxiliary/out/{name}` 下的两张提示图与 `meta.json`）
  ```bash
  python -m scripts.main --stage region_prompts --name q
  ```

- 只跑 构建 prior 与 boxes（读取 `auxiliary/llm_out/{name}_output.json`）
  ```bash
  python -m scripts.main --stage build_prior --name q
  ```
  输出：
  - `auxiliary/box_out/q/q_prior_mask.png`
  - `auxiliary/box_out/q/q_sam_boxes.json`

- 只跑 雕刻阶段（需前两步已完成，且存在 `auxiliary/box_out` 的 prior 与 boxes）
  ```bash
  python -m scripts.main --stage sculpt --name q --text "scorpionfish" --debug
  ```

可选参数（对应 `scripts/main.py`）
- 网格与迭代控制：
  - `--grid-size 3x3`、`--iterations 3`、`--max-points 12`
- 输出目录：
  - `--output-dir pipeline_output`
- 调试与可视化：
  - `--debug`、`--visualize`

二、直接调用辅助脚本 auxiliary/scripts
- 生成区域提示图与 meta.json（文件：`auxiliary/scripts/make_region_prompts.py`）
  ```bash
  python auxiliary/scripts/make_region_prompts.py \
    --name q \
    --image auxiliary/images/q.png \
    --outdir auxiliary/out \
    --rows 9 --cols 9 --line-thickness 3
  ```
  输出目录：`auxiliary/out/q/`，含：
  - `q_vertical_9.png`、`q_horizontal_9.png`、`q_meta.json`

- 由 LLM ids 生成 prior 掩码与 boxes（文件：`auxiliary/scripts/build_prior_and_boxes.py`）
  ```bash
  python auxiliary/scripts/build_prior_and_boxes.py \
    --name q \
    --image auxiliary/images/q.png \
    --meta auxiliary/out/q/q_meta.json \
    --pred auxiliary/llm_out/q_output.json \
    --out auxiliary/box_out
  ```
  输出目录：`auxiliary/box_out/q/`，含：
  - `q_prior_mask.png`、`q_sam_boxes.json`

三、直接跑雕刻 CLI cog_sculpt/cli.py
文件：`cog_sculpt/cli.py`

- 同时使用 prior 掩码作为初始 M0 并用 boxes 推 ROI（推荐）
  ```bash
  python -m cog_sculpt.cli \
    --text "scorpionfish" \
    --image auxiliary/images/q.png \
    --boxes auxiliary/box_out/q/q_sam_boxes.json \
    --prior-mask auxiliary/box_out/q/q_prior_mask.png \
    --name q \
    --use-boxes --use-prior --debug
  ```
- 仅用 prior 掩码（M0 与 ROI 都来自 prior 的 bbox）
  ```bash
  python -m cog_sculpt.cli \
    --text "scorpionfish" \
    --image auxiliary/images/q.png \
    --prior-mask auxiliary/box_out/q/q_prior_mask.png \
    --name q \
    --use-prior --debug
  ```
- 仅用 boxes（ROI 来自 boxes，M0 将由 SAM 框预测）
  ```bash
  python -m cog_sculpt.cli \
    --text "scorpionfish" \
    --image auxiliary/images/q.png \
    --boxes auxiliary/box_out/q/q_sam_boxes.json \
    --name q \
    --use-boxes --debug
  ```

说明与注意
- `cog_sculpt/cli.py` 内部会：
  - 强制加载 Alpha-CLIP（需要 `checkpoints/clip_l14_336_grit_20m_4xe.pth`）
  - 强制加载 SAM（默认路径 `models/sam_vit_h_4b8939.pth`）
  - 输出目录默认为 `sculpt_out/{name or scene}/`（我们通过 `scripts/main.py` 已统一到 `pipeline_output/{name}/`）

四、示例与调试脚本
- 简单示例（文件：`examples/simple_example.py`）
  ```bash
  python examples/simple_example.py
  ```
  功能：
  - 演示 Alpha-CLIP 评分
  - 演示 `SemanticGuidedSAM` 简化版（在 `sam_integration.py` 内）
  - 小型 Cog-Sculpt API demo
  - 输出到 `examples/output/`

- Alpha-CLIP 加载自检（文件：`debug_alpha_clip.py`）
  ```bash
  python debug_alpha_clip.py
  ```
  用途：
  - 快速检查 Alpha-CLIP 模块与权重能否正确加载

五、典型使用顺序（建议）
1) 生成区域提示图与 meta.json
   ```bash
   python -m scripts.main --stage region_prompts --name q
   ```
2) 准备 LLM 输出 ids
   - `auxiliary/llm_out/q_output.json`（你已有示例）
3) 生成 prior + boxes
   ```bash
   python -m scripts.main --stage build_prior --name q
   ```
4) 运行雕刻（Alpha-CLIP × SAM）
   ```bash
   python -m scripts.main --stage sculpt --name q --text "blurred person standing in field" --visualize --debug
   ```
   或一条全流程：
   ```bash
   python -m scripts.main --name q --text "blurred person standing in field" --visualize --debug
   ```

若你想进一步“参数化 SAM 权重路径/类型”，我可以把 `cog_sculpt/cli.py` 中固定的 `models/sam_vit_h_4b8939.pth` 替换成命令行参数（例如 `--sam-ckpt`、`--sam-type`），以便灵活切换。