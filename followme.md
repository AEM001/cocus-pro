已经有了 **ROI 的框 + 初始掩码**。接下来要做的“VLM 驱动语义评估 + 正/负点生成（sculpting）”的关键在于：
**“点坐标由算法采样产生，VLM 只负责给每个候选点的局部 patch 打标签（pos/neg）并给出置信度”**。
下面给你一份**可落地的实现方案**（模块化+提示词+伪代码），照着做就能跑通迭代。

---

# 一、输入 / 输出规范

* **输入**：RGB 图像 `I`，可选深度 `D`；ROI 框 `B=(x0,y0,x1,y1)`（已含 20% padding 更好）；初始掩码 `M0`（与 I 同分辨率）。
* **输出（每一轮）**：

  * 语义评估 `Fsem`（紧凑 JSON，指出“缺哪/多哪/边界如何/关键线索”）；
  * 正负点集合 `P⁺, P⁻`（图像坐标、label 1/0、置信度）；
  * 新掩码 `Mt+1 = SAM(I, [P⁺,P⁻], B)`。

> 迭代 2–3 轮通常够用；每轮点数建议总计 10–16 个（正/负各约 5–8），空间上做去重，避免扎堆。

---

# 二、模块 A：候选点采样器（算法产坐标）

VLM 不生成点坐标；**由我们先生成候选点**，再请 VLM 判别。

在 `B` 内做 **三路并行采样**，合并去重：

1. **内点（mask 内部 → 正点候选）**

   * 对 `M` 做**距离变换**（离边界越远越大），在高响度处做 peak 采样（防止靠边不稳）。
   * 也可在 `M` 内做均匀网格采样，过滤离边界 < r 的点。

2. **边界带（mask 边界 ±带宽 → 双向候选）**

   * 用 `band = dilate(M, k) - erode(M, k)` 得到**边界环**，在环带内均匀取点；这些点经 VLM 决定是补漏（正）还是外溢（负）。

3. **外点（B 内、M 外 → 负点候选）**

   * 在 `B \ M` 做均匀网格 + 纹理相似性筛选（可用局部方差/CLIP 局部相似度，挑选“最像目标但高概率为背景”的区域做难负样）。

> 每一路先各取 20–40 个候选，**合并后做空间 NMS**（半径 `r = 0.06 × diag(B)`），保留**最多 \~30 个总候选**，后续再交给 VLM 分类，最终只取分数最高的若干点用于本轮 SAM。

---

# 三、模块 B：Patch 裁剪与多尺度增强

给 VLM 的是**候选点为中心的小块 patch**（不是整图），并做轻量多尺度以提升稳健性。

* **尺度**：以 `smin = min(wB,hB)`，取相对边长 `{0.08, 0.12, 0.18} × smin` 生成 2–3 个尺度；
* **增强**：保持极轻（可选）——`hflip`、亮度 ±10%、对比度 ±10%（不要重口味，以免伪装线索被破坏）。
* 对每个点得到 `K` 个 patch（K=2\~6），**VLM 逐个判别**，最后对该点做**投票/平均置信**。

---

# 四、模块 C：VLM 语义评估（Peval）与点判定（Pgen）

## C1) 面向掩码的**语义评估 Peval**（每轮一次）

作用：让 VLM **快速指出当前掩码的问题**，为后续点采样做引导（例如“左上边缘外溢”“中心有缺口”“边界发糊”）。

* **输入**：ROI 裁剪图 + 半透明叠加的当前掩码（可在图上画出红色 30% 透明度）+（可选）深度可视化。
* **产出**：紧凑 JSON（不暴露长推理）。

**Peval 提示词（英文示例，强制 JSON）**：

```
SYSTEM: You are a segmentation inspector. Be concise.
USER: You see an ROI crop, with a semi-transparent mask overlay.
Target category: "<instance>" (single target).
Task: Briefly diagnose mask defects. Reply in compact JSON only:
{
  "missing_parts": ["..."],      // where the mask under-covers
  "over_segments": ["..."],      // where it leaks into background
  "boundary_quality": "sharp|soft|fragmented",
  "key_cues": ["shape/texture/depth cues for the true object"]
}
Word budget: within 40 words total.
```

> 用 `missing_parts/over_segments` 的方向词（如 upper-left rim / center core）来**加权下一轮候选点的采样概率**：缺失处增加“内点/边带点”的采样，外溢处增加“边带/外点”的采样。

## C2) 面向 patch 的**点判定 Pgen**（逐点多尺度）

* **输入**：patch 图像（多尺度/增强若干张）、目标类别 `instance`、来自 Peval 的 `key_cues`（8–15 字概括）。
* **输出**：对每个 patch 返回 `is_target` 与 `conf`（0..1），**不输出长解释**。

**Pgen 提示词（英文示例，强制 JSON）**：

```
SYSTEM: You label whether a small image patch belongs to the SINGLE target category.
Be strict and concise. Do not reveal your reasoning.
USER: Target: "<instance>".
Visual cues (may help disambiguation): "<key_cues from Peval in 8-15 words>".
Answer JSON only:
{"is_target": true|false, "conf": 0.00-1.00}
```

* **点级融合**：对同一候选点的多尺度/增强结果取 `p = mean(conf_yes)`（将 false 视为 0）。

* **可选融合**：与 CLIP 文本相似度 `s_clip` 做线性融合 `s = α*p + (1-α)*σ(s_clip)`（α≈0.7，`σ`线性拉伸到 0..1）。

* **阈值自适应**：

  * 随机在 `B` 边缘地带取 20 个“背景对照 patch”，估计负分布 `μ_neg, σ_neg`；
  * 设 `τ_pos = clamp(μ_neg + 2σ_neg, 0.55, 0.7)`；`τ_neg = 0.45`（或对称策略）。
  * `s ≥ τ_pos → 正； s ≤ τ_neg → 负； 中间 → 忽略`。

---

# 五、模块 D：选点策略（从候选到 P⁺/P⁻）

1. 将候选按 `s` 排序；
2. **空间去重**：半径 `r = 0.06 × diag(B)` 的圆形 NMS；
3. 选 **Top-K\_pos（\~5–8）** 为正点，**Bottom-K\_neg（\~5–8）** 为负点；
4. **类型覆盖**：保证既有“内核正点”（mask 内+深处），也有“边界正点”；负点优先取**边界外溢**与**高度背景同构**区域；
5. 坐标从 ROI 局部换回**整图坐标**，SAM 需要 `points` (N,2) 与 `labels` (N,)（1=正，0=负）。

---

# 六、模块 E：SAM 细化与早停

* **推理**：`Mt+1 = SAM.predict(points=P⁺∪P⁻, labels, box=B)`（SAM2 可同时传 bbox 与 points）。
* **早停**：若 `IoU(Mt+1, Mt) < 0.005` 或 `boundary-F` 提升 < 0.3% 连续两轮，或面积/拓扑震荡，提前停止；
* **保底**：保留历轮中**VLM 复核更可信**的一版（例如对“整图/两版叠图”让 VLM 选更好的那一个，或用 CLIP+形状先验做打分）。

---

# 七、伪代码（最小工作流）

```python
# --- Candidate sampling ---
def sample_candidates(M, B, max_total=30, nms_r_ratio=0.06):
    inside = sample_peaks_in_mask(M)                 # 距离变换 + 峰值
    band   = sample_uniform_in_band(M, k=3)          # 膨胀-腐蚀环带
    outside= sample_uniform_outside(M, B)            # B内M外
    cand = merge_and_nms(inside + band + outside, r=nms_r_ratio*diag(B))
    return cand[:max_total]

# --- Patch extraction (multi-scale) ---
def extract_patches(I, D, pts, B):
    scales = [0.08, 0.12, 0.18]  # × min(wB,hB)
    patches = {i: multi_crop(I, pt, B, scales, aug_light=True) for i,pt in enumerate(pts)}
    return patches  # dict: idx -> List[np.ndarray]

# --- Peval: mask-level semantic inspection by VLM ---
def peval_semantics(vlm, I_roi, M_roi, D_roi=None):
    overlay = draw_overlay(I_roi, M_roi)
    prompt = PEVAL_PROMPT(instance, max_words=40)
    return vlm(overlay, D_roi, prompt)  # -> {"missing_parts":[], "over_segments":[], "boundary_quality":"...", "key_cues":[]}

# --- Pgen: per-patch classification by VLM (+ optional CLIP fusion) ---
def classify_points(vlm, patches_dict, key_cues, instance, clip=None, alpha=0.7):
    scores = {}
    for i, patch_list in patches_dict.items():
        vlm_conf = []
        for patch in patch_list:
            resp = vlm(patch, PGEN_PROMPT(instance, key_cues))
            p = resp["conf"] if resp["is_target"] else 0.0
            if clip is not None:
                s_clip = clip_similarity(patch, f"a photo of {instance}")
                p = alpha*p + (1-alpha)*normalize(s_clip)
            vlm_conf.append(p)
        scores[i] = sum(vlm_conf) / len(vlm_conf)
    return scores  # idx -> fused score in [0,1]

# --- Select P+/P- with adaptive thresholds & spatial NMS ---
def select_points(candidates, scores, B, K_pos=6, K_neg=6):
    # 估计负对照阈值
    mu_neg, sigma_neg = estimate_neg_distribution(scores, candidates, B)
    tau_pos = min(max(mu_neg + 2*sigma_neg, 0.55), 0.70); tau_neg = 0.45
    # 正负划分
    pos_pool = [c for c in candidates if scores[c.idx] >= tau_pos]
    neg_pool = [c for c in candidates if scores[c.idx] <= tau_neg]
    pos = spatial_nms_sorted(pos_pool, scores, B, topk=K_pos)
    neg = spatial_nms_sorted(neg_pool, {i:1-scores[i] for i in [n.idx for n in neg_pool]}, B, topk=K_neg)
    # 组织 SAM 输入
    points = np.array([p.xy for p in pos + neg], dtype=np.float32)  # 全图坐标
    labels = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    return points, labels

def sculpt_round(I, D, B, M, sam, vlm, clip=None):
    cand = sample_candidates(M, B)
    patches = extract_patches(I, D, [c.xy for c in cand], B)
    Fsem = peval_semantics(vlm, crop(I,B), crop(M,B), crop(D,B) if D is not None else None)
    scores = classify_points(vlm, patches, key_cues=shorten(Fsem["key_cues"]), instance=INSTANCE, clip=clip)
    P, L = select_points(cand, scores, B)
    M_new = sam.predict(points=P, labels=L, box=B)
    return M_new, Fsem, (P, L)
```

---

# 八、工程落地建议

* **并发 / 速率**：把同一轮的 patch 做**批量**送 VLM（如果接口支持多图）；不支持就分组串行，记得缓存判定结果（坐标+尺度为 key）。
* **可解释日志**：保存 `Peval JSON`、点级 `score`、最终选中 `P⁺/P⁻` 的可视化（小圆点：绿/红）。
* **回退策略**：若 VLM 不稳定，先用 CLIP+深度的简易二阶段门控跑通，再逐步切换 VLM 判定。
* **早停**：IoU/边界-F 收敛 + VLM 在 Peval 的 `missing_parts/over_segments` 稳定为“\[]”。
* **类别提示**：把你在 RSVP `reason` 里“优化过的 summary”压缩成 8–15 个词作为 `key_cues`，提升 VLM 判点的判别力。

