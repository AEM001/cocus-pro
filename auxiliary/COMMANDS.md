# 命令行使用说明（推理-only）

本流程严格按 guide.md 设计，采用相对路径与简单命令行参数，无需 YAML。

目录约定（相对 auxilary/ 运行）
- 输入图片：images/{name}.png（或 .jpg）
- 第一步输出：out/{name}/
- LLM 输出：llm_out/{name}_output.json
- 初始框输出：box_out/{name}/{name}_sam_boxes.json

一、生成区域提示与 meta
- 默认 9x9 网格，线宽 3 像素。
- 输出：
  - out/{name}/{name}_vertical_{rows}.png
  - out/{name}/{name}_horizontal_{cols}.png
  - out/{name}/{name}_meta.json

用法：
- 指定图片名（推荐）：
  python scripts/make_region_prompts.py --name q
- 或显式指定路径：
  python scripts/make_region_prompts.py --image images/q.png --rows 9 --cols 9 --outdir ./out

常用参数：
- --rows, --cols：网格大小，默认 9, 9
- --line-thickness：线宽，默认 3
- --font-size 或 --font-scale：字号，默认使用 --font-scale=0.6 自动估算
- --outdir：输出根目录，默认 ./out

二、准备 LLM 输出
- 将 VLM/LLM 的行/列 ids 结果保存为：
  llm_out/{name}_output.json
- 支持字段（任选其一）
  - ids_line_vertical / ids_line_horizontal
  - ids_vertical / ids_horizontal

示例（q_output.json）：
{
  "ids_line_vertical": [3,4,5],
  "ids_line_horizontal": [5,6,7]
}

三、由 ids 生成初始 SAM 框
- 输入：
  - out/{name}/{name}_meta.json
  - llm_out/{name}_output.json
- 输出：
  - box_out/{name}/{name}_prior_mask.png（黑白掩码，0/255）
  - box_out/{name}/{name}_sam_boxes.json

用法：
- 指定图片名（推荐）：
  python scripts/build_prior_and_boxes.py --name q
- 或显式指定路径：
  python scripts/build_prior_and_boxes.py \
    --image images/q.png \
    --meta out/q/q_meta.json \
    --pred llm_out/q_output.json \
    --out box_out

注意：
- 本脚本仅输出 boxes json，不保存遮罩/叠加图。
- 所有路径均为相对路径，可直接在 auxiliary/ 下运行。

四、常见问题
- 路径不存在：请确认 images/{name}.* 存在，且先运行第 1 步生成 out/{name}/{name}_meta.json。
- JSON 字段不一致：脚本已兼容 ids_line_vertical/horizontal 与 ids_vertical/horizontal。