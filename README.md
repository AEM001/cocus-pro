分割雕刻（VLM驱动）
==================

实现了 `followme.md` 中描述的VLM驱动的语义评估 + 正负点生成（"雕刻"）流水线。

包含内容
--------
- 候选采样（内部/边界/外部），空间非极大值抑制
- 补丁提取（多尺度，轻度数据增强）
- VLM提示和接口（Peval + Pgen）
- Qwen2.5-VL集成脚手架和模拟VLM
- 自适应阈值和NMS的选择策略
- SAM精化钩子和早停（默认返回前一个掩码的存根）
- CLI运行器执行1-N轮雕刻

快速开始（模拟VLM）
------------------

运行单张图像（使用 `auxiliary` 中的现有资源）：

```
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model mock \
  --out_dir outputs/sculpt/dog \
  --rounds 2
```

结果保存在 `outputs/sculpt/dog/` 下（点可视化、peval JSON、分数JSON、每轮掩码、最终掩码）。

使用Qwen2.5-VL-7B-Instruct
--------------------------

- 集成说明请参见 `models/README.md`。
- 默认情况下，`QwenVLM` 在"服务器"模式下运行，但HTTP调用是无操作存根。请实现您的本地服务器或填写 `_infer_via_local`。
- 示例（服务器模式）：

```
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model qwen \
  --server_url http://127.0.0.1:8000/generate \
  --rounds 2
```

SAM集成
-------
- 已接入基于 `segment_anything` 的真实 SAM 推理（如未安装则自动回退为 no-op）。
- 提供检查点与模型类型即可启用：

```
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model mock \
  --sam_checkpoint models/sam_vit_h_4b8939.pth \
  --sam_type vit_h \
  --sam_device cuda \
  --rounds 2
```

- 需要 `torch` 与 `segment_anything`，检查点需与 `--sam_type` 一致（vit_h|vit_l|vit_b）。

注意事项
--------
- 阈值和采样大小遵循 `followme.md`，在代码中易于调整。
- 流水线是模块化的，仅通过文件缓存；如需要可添加记忆化。

---
详细信息
========
完成 — 我实现了VLM驱动的雕刻流水线，并构建了一个干净、模块化的结构，包含Qwen2.5-VL占位符和一个可以端到端运行的工作模拟器。

我添加的内容
------------

scripts/run_sculpt.py: 运行1-N轮雕刻的CLI（Peval + Pgen → P⁺/P⁻ → SAM精化）。
src/sculptor/candidates.py: 内部/边界/外部候选采样，带空间NMS。
src/sculptor/patches.py: 多尺度补丁提取 + 轻度数据增强。
src/sculptor/vlm/prompts.py: 完全按照followme.md的Peval/Pgen提示。
src/sculptor/vlm/base.py: VLM接口 + 健壮的JSON解析。
src/sculptor/vlm/qwen.py: Qwen2.5-VL存根（服务器/本地模式；默认无网络）。
src/sculptor/vlm/mock.py: 用于本地测试的确定性模拟VLM。
src/sculptor/select_points.py: P⁺/P⁻的阈值 + 空间NMS选择。
src/sculptor/sam_refine.py: SAM集成钩子 + 早停（IoU）回退。
src/sculptor/utils.py, src/sculptor/types.py: IO/可视化助手和类型。
models/README.md: 如何插入您的本地Qwen模型/服务器。
README.md: 快速使用和集成说明。

如何运行（模拟）
---------------

命令：
scripts/run_sculpt.py:1
示例：
python scripts/run_sculpt.py --image auxiliary/images/dog.png --mask auxiliary/box_out/dog/dog_prior_mask.png --roi_json auxiliary/box_out/dog/dog_sam_boxes.json --instance "dog" --model mock --out_dir outputs/sculpt/dog --rounds 2
输出：
outputs/sculpt_test/dog/dog/round0_points.png
outputs/sculpt_test/dog/dog/round0_peval.json
outputs/sculpt_test/dog/dog/round0_scores.json
outputs/sculpt_test/dog/dog/final_mask.png

Qwen集成
--------

代码脚手架：
src/sculptor/vlm/qwen.py:1
src/sculptor/vlm/prompts.py:1
选项：
服务器模式：实现您的本地HTTP端点并传递--server_url或设置QWEN_VL_SERVER。
本地模式：实现_infer_via_local并传递--model_dir或设置QWEN_VL_MODEL_DIR。
文档：models/README.md:1

SAM钩子
-------

入口：src/sculptor/sam_refine.py:1
当前：安全回退（返回前一个掩码）。在标注的地方注入您自己的预测器（SAM/SAM2）以实际使用点和标签进行精化。

关键文件
--------

流水线：
src/sculptor/candidates.py:1
src/sculptor/patches.py:1
src/sculptor/select_points.py:1
src/sculptor/sam_refine.py:1
提示/模型：
src/sculptor/vlm/prompts.py:1
src/sculptor/vlm/qwen.py:1
src/sculptor/vlm/mock.py:1

说明
----

采样/阈值匹配followme.md默认值（尺度{0.08, 0.12, 0.18}，NMS r=0.06×diag(B)，K_pos/K_neg可调）。
Peval的"missing_parts/over_segments"已记录并准备指导未来的采样权重；当前采样器是几何驱动的（可扩展为基于Peval线索偏向区域）。
如果跳过负控制补丁，阈值回退到安全启发式；如需要，您可以在select_points.py中添加可选的背景控制评分。

您希望我：

现在连接Qwen服务器调用或实现进程内Qwen加载器？
连接真正的SAM/SAM2预测器进行实际掩码精化？
