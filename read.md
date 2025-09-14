与项目其它模块的关系

alpha_clip_rw/alpha_clip.py
：提供底层的模型加载、RGBA/掩码预处理、文本分词。
cog_sculpt/core.py：在语义雕刻迭代过程中，会基于 
score_region_with_templates()
 的评分对网格区域挑选正/负点，并驱动 sam_integration.py 做交互式细化（对应 CLAUDE.md 中 6 步流程的第 3、4 步）。
sam_integration.py：实现与 SAM/SAM2 的统一交互封装，消费这里输出的语义分数来指导点选。
为精确解释 Alpha 通道与 RGB 如何在模型里共同起作用，我将查看 `alpha_clip_rw/model.py` 中视觉分支的实现细节，看看 `model.visual(image_tensor, alpha_tensor)` 如何融合两者。

我将打开 [alpha_clip_rw/model.py](cci:7://file:///Users/Mac/code/cocus-pro/Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:0:0-0:0)，查看视觉分支 `visual` 的前向实现，明确 Alpha 通道在 forward 中与 RGB 特征融合的具体方式，这样能给你一份基于源码的解释。

下面把“Alpha 通道如何与 RGB 一起起作用”讲清楚，给你源码级定位、数据流与实操建议。

一、融合发生在什么位置（源码指向）
- ResNet 视觉分支
  - 文件与位置：[alpha_clip_rw/model.py](cci:7://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:0:0-0:0) -> [ModifiedResNet.forward()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:44:4-57:18) 的 [stem()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:143:8-148:20)。
  - 关键代码：
    - `x = self.relu1(self.bn1(self.conv1(x) + self.conv1_alpha(alpha)))`（行 145）
  - 含义：
    - 对 RGB 做一次 3×3/stride=2 卷积 `conv1(x)`。
    - 对 Alpha 单通道做一次 3×3/stride=2 卷积 `conv1_alpha(alpha)`。
    - 两者逐元素相加，再进入后续层。
    - 直观理解：Alpha 的卷积特征作为“加性偏置”注入，强调/抑制某些空间区域的视觉响应。

- ViT 视觉分支
  - 文件与位置：[alpha_clip_rw/model.py](cci:7://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:0:0-0:0)
    - [VisionTransformer.forward()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:44:4-57:18) 行 585-589
    - [VisionTransformer_MaPLe.forward()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/model.py:44:4-57:18) 行 528-533
  - 关键代码（两者逻辑一致）：
    - `x = self.conv1(x)`（将 RGB 切成 patch，投到通道维）
    - `x = x + self.conv1_alpha(alpha)`（将 Alpha 也切成同样 patch 后与 RGB patch 特征相加）
  - 含义：
    - 对 RGB 与 Alpha 分别做 patch embedding（Conv2d 实现），然后相加。
    - 直观理解：每个 patch 的 token 特征 = RGB_token + Alpha_token。Alpha 直接改变 token 的初始表征，从而影响 Transformer 后续的注意力与最终 pooled 特征。

二、数据流与张量规范（推理端）
- 入口
  - [alpha_clip_rw/inference.py](cci:7://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/inference.py:0:0-0:0) -> [encode_image_rgba()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/inference.py:152:4-190:29)（行 150-189）
  - 预处理：
    - RGB：`self.preprocess(image_pil)`，源自 [alpha_clip.load()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:113:0-226:59) 返回的标准 CLIP 预处理（[_transform()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:85:0-92:6)）。
    - Alpha：`self.mask_preprocess(alpha_pil)`，定义于 `alpha_clip_rw/alpha_clip.py:88-94` 的 [mask_transform()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:94:0-100:6)，包含 `Resize -> CenterCrop -> ToTensor -> Normalize(0.5, 0.26)`。
  - 前向：
    - `self.model.visual(image_tensor, alpha_tensor)`（行 183-186），把两路张量都传入视觉分支。

- 形状与取值
  - 经过预处理后，模型期望的形状：
    - `image_tensor`: `[B, 3, H', W']`
    - `alpha_tensor`: `[B, 1, H', W']`
    - 其中 `H' = W' = self.input_resolution`（比如 336）。
  - 你的 Alpha 掩码输入可以是：
    - `[H, W]` numpy 或 PIL 的 L 模式，值域 [0,1] 或 [0,255] 都行。
    - 代码里如果检测到 `alpha.max() <= 1.0` 会先放大到 [0,255] 再转 PIL，之后交给 [mask_transform](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:94:0-100:6) 做统一规范化。

- 注意：ViT 的 forward 里有注释“ASSUME alpha is always not None!”（行 531、587），因此使用 ViT 结构时必须给 Alpha；ResNet 版本也在 stem 里显式用到了 `conv1_alpha(alpha)`。

三、为什么这样融合有用（直觉和效果）
- 加性注入的直觉
  - 在第一层或 patch embedding 处把 Alpha 的“注意力先验”加到视觉特征上，意味着“被 Alpha 高亮的地方”在后续网络中更容易产生更强的响应。
  - 这比后处理的 mask 更早、更“软”地影响特征提取，能更细腻地引导局部表征。
- 对雕刻流程的意义
  - 你传入的 Alpha（通常来自当前候选区域/初始掩码/先验）会让模型“更关注这个区域”，从而提升该区域与目标文本的匹配置信度；反之，Alpha 低的地方贡献相对被抑制。
  - 配合多模板评分做聚合，能更稳健地评估“这个候选区域是不是目标”。

四、Alpha 应该表示什么（怎么构造更好）
- 表达“该区域的重要度/关注度”，0 表示不关注，1 表示强关注。最佳实践：
  - 从 box 得到粗掩码：框内=1，外=0，再对边界做高斯模糊/距离变换，形成软边界。
  - 从现有分割 mask 出发：把目标区域设为高值，背景设为低值，同样做一点 blur，使其更“软”。
  - 多轮迭代时：可把上一次的预测作为下一轮的 Alpha 起点，逐步细化（这与“雕刻”的迭代精神一致）。

五、常见坑点与排查
- 尺寸对齐
  - 不需要你手工对齐，但务必传入与图像同分辨率的 Alpha；[mask_transform](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:94:0-100:6) 会做同样的 `Resize/CenterCrop` 与规范化，保证与 RGB 经过相同的几何路径。
- 值域与类型
  - 0~1 或 0~255 都支持；内部会统一。若你自己做了归一化，不要重复把值再标准化两次。
- None 问题
  - 使用 [encode_image_rgba()](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/inference.py:152:4-190:29) 就别传 `alpha=None`；ViT 分支假定 alpha 存在。
- 预处理一致性
  - 避免单独对 Alpha 做与 [mask_transform](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/alpha_clip.py:94:0-100:6) 不一致的几何变换（如额外裁剪/缩放），否则会导致与 RGB 的空间错位。

六、和“主干权重 vs Alpha 分支权重”的关系
- 主干权重（CLIP backbone）
  - 定义模型大结构与基础表征能力（文本/图像编码器）。
  - 来源：官方 CLIP 权重（如 `ViT-L/14@336px`）。加载代码：`alpha_clip_rw/alpha_clip.py:135-161`。
- Alpha 分支权重
  - 定义 `conv1_alpha` 这条 Alpha 支路（以及可能的适配参数）的权重，让模型“学会理解 Alpha 的语义意义”。
  - 加载时机：`alpha_clip_rw/alpha_clip.py:158-161`，将权重加载到 `model.visual`，与主干一起工作。
  - 直观理解：CLIP backbone 决定“怎么看图/懂文本”，Alpha 分支权重让它“懂得用一张软掩码来调整看图的方式”。

七、一个小对比实验（帮助形成手感）
- 用完全全黑的 Alpha（全 0）与“目标区域为 1、背景为 0”的 Alpha，分别计算 [compute_similarity](cci:1://file:///Users/Mac/code/cocus-pro/alpha_clip_rw/inference.py:192:4-227:39)：
  - 期望现象：有 Alpha 引导时，目标类别 prompt 的分数会更高，非目标会更低；全 0 的 Alpha 则接近“抑制一切区域”的极端情况（通常分数不理想）。
- 代码片段
```python
from alpha_clip_rw.inference import AlphaCLIPInference
import numpy as np

infer = AlphaCLIPInference(
    model_name="ViT-L/14@336px",
    alpha_vision_ckpt_pth="checkpoints/clip_l14_336_grit_20m_4xe.pth",
    device="cuda"  # 或 "cpu"
)

rgb = np.asarray(...)      # H,W,3
alpha_zero = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)        # 全 0
alpha_soft = np.asarray(...)  # 你的候选区域软掩码，0~1

txts = ["a photo of the scorpionfish."]

s0 = infer.compute_similarity(rgb, txts, alpha=alpha_zero)   # 基线
s1 = infer.compute_similarity(rgb, txts, alpha=alpha_soft)   # 有区域引导
print("no-alpha score:", float(s0.squeeze()))
print("soft-alpha score:", float(s1.squeeze()))
```

core.py
角色：实现“语义雕刻 + SAM”的核心流程，包括网格划分、区域语义评分、正负点选择、SAM 交互细化与迭代早停。

在cli.py中，是用来设定参数的，主要提供text，boxes.json, 初始的mask

---
# 点提示实现
下面给你精确定位“点提示如何起作用”，以及“它与上一步掩码如何共同工作”的源码位置与数据流。

一、点提示真正起作用的地方
- 文件/类/方法：[sam_integration.py](cci:7://file:///Users/Mac/code/cocus-pro/sam_integration.py:0:0-0:0) -> [SAMWrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)（行 121-152）
  - 入口参数：
    - `point_coords: np.ndarray`，形状 [N, 2]，坐标格式为 `[x, y]`
    - `point_labels: np.ndarray`，形状 [N]，1 表示正点、0 表示负点
  - 核心调用：
    - 调用了 `self.predictor.predict(point_coords=..., point_labels=..., multimask_output=True)`
    - 返回多个候选掩码 `masks` 与分数 `scores`，然后选 `np.argmax(scores)` 对应的掩码作为输出
  - 注意：
    - 必须先调用 [set_image(image)](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:87:4-95:34)（行 88-97），否则会报错（行 136-138）
    - 若点数为 0，直接返回全 0 掩码（行 139-143）
- 同构的 SAM2 实现：[SAM2Wrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)（行 198-219），逻辑一致，仅底层 predictor 不同。

小结：点提示对 SAM 的作用就是通过 `predictor.predict(point_coords, point_labels)` 触发交互预测，SAM 根据这些点（正/负）在当前图上输出一个新掩码。

二、“与上一步掩码一起”是如何发挥作用的
- 关键点：SAM 的 [predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 本身不接收“上一步的掩码”作为输入。它只看“当前图像 + 点提示”。“上一步掩码”的作用体现在迭代流程中，用于帮助选择新一轮的点，然后用这些点去获得更精细的新掩码。
- 两种体现位置：
  1) [cog_sculpt/core.py](cci:7://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:0:0-0:0) -> [sculpting_pipeline()](cci:1://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:530:0-636:12)（行 449-540）
     - 前一轮得到的掩码 `M` 用来构造每个网格单元的 Alpha（[build_alpha_for_cell()](cci:1://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:225:0-230:16)），交给 [RegionScorer](cci:2://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:255:0-258:33)（通常是 [AlphaCLIPScorer](cci:2://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:261:0-286:22)）打分（行 472-477）
     - 基于阈分选出正/负单元格，调用 [select_points_from_cells()](cci:1://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:361:0-407:27) 生成本轮的正负点（行 508-516）
     - 用这些点喂给 [sam.predict_points(image, Ppos, Pneg)](cci:1://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:467:4-506:51)（即 [SAMWrapper.predict_with_points](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)），得到 `M_new`（行 532）
     - 用 IoU 做早停或进入下一轮（行 534-537）
     - 也就是说：上一步掩码只是“点的选择依据”，并不作为 SAM 的输入；真正进入 SAM 的，是点提示本身
  2) [sam_integration.py](cci:7://file:///Users/Mac/code/cocus-pro/sam_integration.py:0:0-0:0) -> [SemanticGuidedSAM.segment_with_semantic_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:242:4-313:27)（行 243-314）
     - 初始化 [current_mask = self.sam.predict_with_box(bbox)](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:34:4-37:12)（行 271-273）
     - 用当前 `current_mask` 网格化生成 Alpha，调用 Alpha-CLIP 评分后生成正负点（[_generate_semantic_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:315:4-373:37)，行 316-375）
     - 将点转换为 `[x, y]` 格式与标签 1/0，调用 [self.sam.predict_with_points(...)](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 得到 `new_mask`（行 300-306）
     - 计算 IoU 收敛，否则将 `current_mask = new_mask` 继续迭代（行 308-313）
     - 同样逻辑：mask 只影响“点的挑选”，不直接作为 [predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 的输入

三、点提示数据的组织与坐标/标签规范
- 坐标格式：
  - SAM 需要 `[x, y]`；在 [core.py](cci:7://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:0:0-0:0) 中生成的点是 [Point(y, x, is_positive)](cci:2://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:63:0-74:22)，在 [SamWrapper.predict_points()](cci:1://file:///Users/Mac/code/cocus-pro/cog_sculpt/core.py:467:4-506:51) 内部做了 `(y,x) -> (x,y)` 的转换（`cog_sculpt/core.py:407-423`），组装 `point_coords` 与 `point_labels`
  - 在 [sam_integration.py](cci:7://file:///Users/Mac/code/cocus-pro/sam_integration.py:0:0-0:0) 的 [SemanticGuidedSAM.segment_with_semantic_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:242:4-313:27) 里也明确把 `(y, x)` 转为 `[x, y]` 再传入（行 292-298）
- 标签：
  - 1=正点，0=负点（[SAMWrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 行 130-132）

四、你要看“点提示如何与上一轮结果结合”的最短源码跳转
- 点提示触发分割（真正作用点）
  - `sam_integration.py:121-152` [SAMWrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)
  - 或 `sam_integration.py:198-219` [SAM2Wrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)
- 上一轮掩码如何用于生成点
  - `cog_sculpt/core.py:472-477` 评分
  - `cog_sculpt/core.py:508-516` 阈分与点选择
  - `cog_sculpt/core.py:532` 用点进行 SAM 预测更新掩码
- 简化集成版（单文件示例）
  - `sam_integration.py:316-375` [_generate_semantic_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:315:4-373:37)（由 `current_mask` 网格化构造 Alpha，并经 Alpha-CLIP 打分后选点）

五、如果你想“把上一轮掩码也传给 SAM 作为条件”的思路
- 现有 `SamPredictor.predict()` 接口不直接接收“上一轮掩码”作为输入条件（它有 `mask_input` 的接口用于细节变体，但标准使用是点/框/多掩码输出）。
- 目前本项目采用的是“mask 只影响点选择”的迭代范式。如果要探索“把 mask 直接传给 SAM 作为额外提示”，需要切换到 SAM 的低层 API 或实现自定义策略（例如以 mask 轮廓采样点、一致性点等）。

总结
- 点提示对分割的直接作用在 [SAMWrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12)/[SAM2Wrapper.predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 中，调用 `predictor.predict(point_coords, point_labels)` 返回掩码。
- “上一轮的掩码”并不直接作为 [predict_with_points()](cci:1://file:///Users/Mac/code/cocus-pro/sam_integration.py:39:4-46:12) 的输入，而是用于“如何选点”的逻辑，从而间接推动下一轮更精细的分割。

---
