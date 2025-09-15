● 📋 精细雕刻阶段完整流程详解

  基于代码分析，这里是精细雕刻阶段的详细流程，按文件和函数逐一说明：

  ---
  🎯 阶段1：初始状态准备

  文件：scripts/run_sculpt_simple.py

  函数：main() 第198-205行
  # Step 1: 生成初始SAM掩码
  current_mask = generate_sam_mask_from_prior(I, B, sam, prior_mask)
  - 输入：边界框(B) + 原始图像(I) + 先验掩码
  - 输出：SAM生成的初始掩码(current_mask)
  - 关键点：掩码严格裁剪到ROI范围内

  ---
  🎯 阶段2：候选点智能采样

  文件：src/sculptor/candidates.py

  函数：sample_candidates() 第37-67行
  cand = sample_candidates(current_mask, B, max_total=30, nms_r_ratio=0.06)
  详细流程：
  1. 边界带计算 (_compute_band() 第103-118行)
    - 使用3像素宽的边界带(k=3)
    - 基于掩码bbox计算，而非ROI
  2. 采样配额分配 (第44-48行)
    - 边界点: 65% (用于边界精细化)
    - 内部点: 10% (验证正确区域)
    - 外部点: 25% (验证排除区域)
  3. 空间NMS去重 (第61-64行)
    - 基于掩码bbox对角线的6%作为NMS半径

  ---
  🎯 阶段3：多尺度上下文提取

  文件：src/sculptor/patches.py

  函数：multi_scale_with_aug() 第216-224行
  patches = multi_scale_with_aug(
      I, pts, B,
      scales=[0.05, 0.15, 0.25],  # 边界敏感的较小尺度
      context=1.15  # 包含边界两侧上下文
  )
  技术细节：
  - 尺度计算：基于掩码实际尺寸，而非ROI
  - 上下文系数：1.15倍确保覆盖边界两侧
  - 增强策略：轻量级水平翻转+亮度微调

  ---
  🎯 阶段4：VLM智能分析

  文件：src/sculptor/vlm/qwen.py 和 base.py

  两个并行分析：

  4.1 整体评估 (Peval)

  Fsem = vlm.peval(overlay, instance=config['instance'])
  - 作用：评估当前掩码整体质量
  - 输出：缺陷诊断 + 关键视觉线索

  4.2 局部评估 (Pgen)

  resp = vlm.pgen(patch, config['instance'], cues)
  - 作用：评估每个候选点是否在目标上
  - 输出：置信度分数 [0.0, 1.0]

  ---
  🎯 阶段5：控制点精选

  文件：src/sculptor/select_points.py

  函数：select_points() 第254行
  P, L, pos, neg = select_points(cand, scores, B, K_pos=6, K_neg=6)
  算法流程：
  1. 自适应阈值：基于分数分布动态确定
  2. 空间NMS：确保控制点分布均匀
  3. 分层选择：高置信度优先，边界重点

  ---
  🎯 阶段6：SAM精细雕刻

  文件：src/sculptor/sam_refine.py

  函数：SamPredictorWrapper.predict() 第266-277行
  new_mask = sam.predict(I, P, L, B, prev_mask=current_mask)
  关键机制：
  - 迭代优化：基于控制点逐步调整边界
  - 早停判断：IoU变化<0.5%时停止
  - 容错处理：无控制点时跳过优化

  ---
  📊 完整数据流总结

  边界框(B) → SAM初始掩码 → 候选点采样 → 多尺度patch提取 →
  VLM双模式分析 → 控制点选择 → SAM迭代精炼 → 最终精细掩码

  🔧 可调参数位置

  | 参数类型    | 位置                       | 默认值              | 作用       |
  |---------|--------------------------|------------------|----------|
  | 边界带宽度   | candidates.py:41         | k=3              | 控制边界探索范围 |
  | 采样总数    | run_sculpt_simple.py:211 | max_total=30     | 控制计算量    |
  | Patch尺度 | run_sculpt_simple.py:220 | [0.05,0.15,0.25] | 控制上下文范围  |
  | 控制点数量   | run_sculpt_simple.py     | K_pos=6, K_neg=6 | 控制SAM输入  |
  | 早停阈值    | sam_refine.py:271        | iou_tol=0.005    | 控制收敛精度   |

  这个流程体现了**"SAM粗分 + VLM精雕 + 边界重点"**的核心设计思想，每个阶段都有明确的输入输出和可调参数。