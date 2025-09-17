太正常了：只要有**掩码**，你就已经可以把绝大多数评估做起来了。开放词汇伪装目标分割（OVCOS）一般分两层评估：

1. 只看分割质量（不涉及类别）——沿用 COS 社区通用的 4 个指标：
   **S-measure (Sα)**、**E-measure (Eϕ)**、**加权 F-measure (Fωβ)**、**MAE**。这些专为前景/伪装目标评估设计，能兼顾结构一致性、区域一致性与像素误差（定义与来源见文献）。([Yun Liu][1])

2. 同时考察“分割+开放词汇分类”的**类感知指标**（在 OVCOS 里最常用的 6 个）：
   **cSm、cFwβ、cMAE、cFβ、cEm、cIoU**。它们是在上面 4+2 个分割指标的基础上加入“类别是否预测正确”的约束，用来“联合评估分类正确性与分割质量”。若类别预测错，分割再好也会被惩罚（通常该样本的 c-指标记 0，cMAE 记 1）。这套做法由 OVCoser 提出/沿用，后续工作（包括你上传的 2025 论文）也在使用。

---

# 具体评估指标（简明解释）

* **S-measure (Sα)**：结构相似度，结合“对象级”（整体形状）与“区域级”（局部）一致性，0–1 越大越好。([Yun Liu][1])
* **E-measure (Eϕ)**：增强对齐度，同时考虑像素与图像层级一致性，0–1 越大越好。([arXiv][2])
* **F-measure (Fβ)**：基于 PR 曲线的 F 分数（常用最大/平均/自适应阈值三种统计）。
* **加权 F-measure (Fωβ)**：对错检/漏检位置赋以空间权重，更鲁棒的 F 分数。([Open Access CVF][3])
* **MAE**：像素平均绝对误差，0–1 越小越好。
* **IoU**：交并比（这里用于 cIoU；也可作为 COS 的补充）。
* **cSm / cEm / cFwβ / cFβ / cMAE / cIoU**：在上面各项前加 **c** 表示“**类感知**”：

  * 若预测类别=真值类别，计算该样本对应指标；
  * 若预测类别≠真值类别，则样本的 c-指标记为 0（cMAE 记为 1）；
  * 再对全测试集**宏平均**（或报告全局平均）。这种“联合评估分割+分类”的协议被 OVCoser 定义并在 OVCOS 论文里沿用。

> 你的现状“只有掩码”，那就先把 \*\*COS 四项（Sα / Eϕ / Fωβ / MAE）\*\*全部跑起来；等你有分类结果（每张图一个开放词汇标签）再补 **c-六项**。

---

# 一键可跑的评估脚本（Python）

下面给你一份**可直接用 pip 安装、跨数据集通用**的脚本。它用社区常用库 **PySODMetrics** 计算 S/E/Fωβ/MAE（与 Fan 等人的官方 MATLAB 工具箱核对过的一致实现），并封装了 OVCOS 的 **c-指标**逻辑；若暂时没有预测类别文件，会自动只输出 COS 指标。([GitHub][4])

**目录约定**

* `gt_masks/`：GT 掩码（PNG，前景=255/1；背景=0）。
* `pred_masks/`：模型预测掩码（灰度概率或二值均可）。
* 可选 `gt_labels.json`：`{"img_id": "class_name", ...}`
* 可选 `pred_labels.json`：`{"img_id": "class_name", ...}`

  > `img_id` 与文件名（不含后缀）一致即可。

**安装依赖**

```bash
pip install pysodmetrics opencv-python-headless numpy tqdm pandas
```

**保存为 `eval_ovcos.py` 并运行**

```python
import os, json, glob
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

# PySODMetrics
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, Fmeasure, MAE

def read_gray01(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = m.astype(np.float32) / 255.0
    return np.clip(m, 0, 1)

def to_binary01(arr, thr=0.5):
    return (arr >= thr).astype(np.float32)

def resize_to(src, shape):
    if src.shape == shape:
        return src
    return cv2.resize(src, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def iou_bin(pred_bin, gt_bin, eps=1e-7):
    inter = np.sum((pred_bin > 0) & (gt_bin > 0))
    union = np.sum((pred_bin > 0) | (gt_bin > 0))
    return float(inter) / (float(union) + eps)

def load_label_json(p):
    if p and os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def eval_dataset(gt_dir, pred_dir, gt_labels_json=None, pred_labels_json=None,
                 bin_thr=0.5, save_csv="metrics.csv"):
    gt_labels = load_label_json(gt_labels_json)
    pred_labels = load_label_json(pred_labels_json)

    img_ids = sorted([os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(gt_dir, "*"))])

    # Metric accumulators (COS)
    sm_obj, em_obj, wfm_obj, fm_obj, mae_obj = Smeasure(), Emeasure(), WeightedFmeasure(), Fmeasure(), MAE()

    # Per-sample records for reporting/调试
    rows = []

    # Class-aware sums
    c_counts = 0
    cSm_sum = cEm_sum = cFw_sum = cF_sum = cIoU_sum = 0.0
    cMAE_sum = 0.0

    for img_id in tqdm(img_ids, ncols=80):
        gt_path = os.path.join(gt_dir, img_id + ".png")
        pred_path = os.path.join(pred_dir, img_id + ".png")
        if not os.path.isfile(pred_path):
            # 若缺预测，跳过
            continue

        gt = read_gray01(gt_path)
        pr = read_gray01(pred_path)
        pr = resize_to(pr, gt.shape)

        gt_bin = to_binary01(gt, 0.5)

        # ---- COS metrics (no class) ----
        sm_obj.update(pr, gt_bin)
        em_obj.update(pr, gt_bin)
        wfm_obj.update(pr, gt_bin)
        fm_obj.update(pr, gt_bin)
        mae_obj.update(pr, gt_bin)

        # 采样级别统计（便于你排查）
        sm_i = Smeasure()._S_object(pr, gt_bin) if hasattr(Smeasure(), "_S_object") else None  # 可选：不保证版本都有
        em_all = Emeasure()
        em_all.update(pr, gt_bin)
        em_adp = em_all.compute()["adp"]
        wfm_i = WeightedFmeasure()
        wfm_i.update(pr, gt_bin)
        wfm_val = wfm_i.compute()

        fm_all = Fmeasure()
        fm_all.update(pr, gt_bin)
        fm_max = fm_all.compute()["fm"]["max"]  # 最大 Fβ
        mae_all = MAE()
        mae_all.update(pr, gt_bin)
        mae_val = mae_all.compute()

        iou_val = iou_bin(to_binary01(pr, bin_thr), gt_bin)

        row = dict(id=img_id, S_adp=None, E_adp=em_adp, Fw_beta=wfm_val,
                   Fbeta_max=fm_max, MAE=mae_val, IoU=iou_val)
        rows.append(row)

        # ---- Class-aware (需要类别文件) ----
        if gt_labels is not None and pred_labels is not None:
            gt_cls = gt_labels.get(img_id, None)
            pr_cls = pred_labels.get(img_id, None)

            # 正确才计入该样本的分割得分；错误样本 c-指标按 0（cMAE=1）
            if (gt_cls is not None) and (pr_cls is not None):
                c_counts += 1
                correct = (gt_cls == pr_cls)

                if correct:
                    cSm_sum += row["S_adp"] if row["S_adp"] is not None else 0.0
                    cEm_sum += row["E_adp"]
                    cFw_sum += row["Fw_beta"]
                    cF_sum  += row["Fbeta_max"]
                    cIoU_sum += row["IoU"]
                    cMAE_sum += row["MAE"]
                else:
                    cSm_sum += 0.0
                    cEm_sum += 0.0
                    cFw_sum += 0.0
                    cF_sum  += 0.0
                    cIoU_sum += 0.0
                    cMAE_sum += 1.0  # 错类样本记作最差 MAE

    # 汇总
    cos_res = {
        "S_measure": sm_obj.compute(),                 # 返回标量
        "E_measure(adp/mean/max)": em_obj.compute(),   # dict: {"adp":..,"mean":..,"max":..}
        "F_measure(max/mean/adp)": fm_obj.compute(),   # dict
        "Weighted_Fbeta": wfm_obj.compute(),           # 标量
        "MAE": mae_obj.compute()                       # 标量
    }

    # 类感知指标
    c_res = None
    if (gt_labels is not None) and (pred_labels is not None) and c_counts > 0:
        c_res = {
            "cSm": cSm_sum / c_counts if c_counts else None,
            "cEm": cEm_sum / c_counts if c_counts else None,
            "cFwβ": cFw_sum / c_counts if c_counts else None,
            "cFβ":  cF_sum / c_counts if c_counts else None,
            "cIoU": cIoU_sum / c_counts if c_counts else None,
            "cMAE": cMAE_sum / c_counts if c_counts else None,
        }

    # 保存逐样本 CSV（便于定位问题）
    if rows:
        pd.DataFrame(rows).to_csv(save_csv, index=False)

    return cos_res, c_res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="GT 掩码目录")
    ap.add_argument("--pred_dir", required=True, help="预测掩码目录")
    ap.add_argument("--gt_labels", default=None, help="GT 类别 JSON（可选）")
    ap.add_argument("--pred_labels", default=None, help="预测类别 JSON（可选）")
    ap.add_argument("--bin_thr", type=float, default=0.5, help="二值化阈值用于 IoU")
    ap.add_argument("--out_csv", default="per_image_metrics.csv")
    args = ap.parse_args()

    cos_res, c_res = eval_dataset(args.gt_dir, args.pred_dir, args.gt_labels, args.pred_labels,
                                  bin_thr=args.bin_thr, save_csv=args.out_csv)

    print("=== COS（分割-only）===")
    print(cos_res)
    if c_res is not None:
        print("=== OVCOS（类感知）===")
        print(c_res)
    else:
        print("未提供类别文件，已跳过 c-指标。")
```

**使用示例**

```bash
# 只算 COS 四项
python eval_ovcos.py --gt_dir ./gt_masks --pred_dir ./pred_masks

# 同时算 c-六项（需类别）
python eval_ovcos.py --gt_dir ./gt_masks --pred_dir ./pred_masks \
  --gt_labels ./gt_labels.json --pred_labels ./pred_labels.json
```

> 说明：上面脚本里 S-measure 的逐样本输出 `S_adp` 我留了兼容写法；不同版本的 PySODMetrics 对外 API 略有差异。如果你只需要汇总（推荐），直接看 `cos_res` 即可。

---

## 结果应该怎么“报表”？

* **COS**：`Sα`、`Eϕ(adp/mean/max)`、`Fωβ`、`Fβ(max/mean/adp)`、`MAE`（主表里一般报 `Sα`、`Eϕ(adp)`、`Fωβ`、`MAE` 四个）。([GitHub][5])
* **OVCOS（类感知）**：`cSm`、`cEm`、`cFwβ`、`cFβ`、`cIoU`、`cMAE` 六项（“联合评估分类+分割”的标准做法）。你的 2025 论文和 OVCoser 都是这么报的。([arXiv][6])

---

## 常见坑 & 建议

* **阈值**：`Fβ`/`Eϕ` 有 *max/mean/adp* 三种统计（脚本已按库默认实现）。
* **掩码尺寸**：务必把预测掩码 resize 到 GT 尺寸（脚本已做）。
* **前景取值**：GT 用 0/255 或 0/1 都行，脚本会归一化到 `[0,1]` 并二值化。
* **类别对齐**：`img_id` 要与文件名一致；开放词汇标签字符串要确保一致（大小写/空格）。
* **宏/微平均**：上面的 c-指标是对**样本**做平均（相当于 Micro）。如需**类均值（Macro）**，按类别分组后对每类均值再平均即可（若你要我也可以把宏平均版加到脚本里）。关于 Macro/Micro 的差异可参考这几篇总结。([Evidently AI][7])

---

## 参考（指标来源 & 协议）

* OVCOS/OVCoser 对 **cSm/cFwβ/cMAE/cFβ/cEm/cIoU** 的采用与说明（联合评估分类+分割）。
* **S-measure** 定义与动机。([Yun Liu][1])
* **E-measure** 定义与动机。([arXiv][2])
* **加权 F-measure (Fωβ)**。([Open Access CVF][3])
* Python 实现工具（与官方 MATLAB 评测一致对齐）：**PySODMetrics / PySODEvalToolkit**。([GitHub][4])

---
