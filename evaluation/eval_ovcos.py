import os, json, glob
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, Fmeasure, MAE

# Adapters for current py_sod_metrics version
class SAdapter:
    def __init__(self): self.m = Smeasure()
    def update(self, pr, gt): self.m.step(pr*255, gt*255)
    def compute(self):
        res = self.m.get_results()
        if isinstance(res, dict):
            # try common keys
            return float(res.get('sm', list(res.values())[0]))
        return float(res)
class EAdapter:
    def __init__(self): self.m = Emeasure()
    def update(self, pr, gt): self.m.step(pr*255, gt*255)
    def compute(self): return self.m.get_results()
class WFMAdapter:
    def __init__(self): self.m = WeightedFmeasure()
    def update(self, pr, gt): self.m.step(pr*255, gt*255)
    def compute(self): return self.m.get_results()
class FAdapter:
    def __init__(self): self.m = Fmeasure()
    def update(self, pr, gt): self.m.step(pr*255, gt*255)
    def compute(self): return self.m.get_results()
class MAEAdapter:
    def __init__(self): self.m = MAE()
    def update(self, pr, gt): self.m.step(pr*255, gt*255)
    def compute(self):
        res = self.m.get_results()
        if isinstance(res, dict):
            return float(list(res.values())[0]) if res else 0.0
        return float(res)

def read_gray01(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(path)
    m = m.astype(np.float32) / 255.0
    return np.clip(m, 0, 1)

def to_binary01(arr, thr=0.5):
    return (arr >= thr).astype(np.float32)

def resize_to(src, shape):
    if src.shape == shape: return src
    return cv2.resize(src, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def iou_bin(pred_bin, gt_bin, eps=1e-7):
    inter = np.sum((pred_bin > 0) & (gt_bin > 0))
    union = np.sum((pred_bin > 0) | (gt_bin > 0))
    return float(inter) / (float(union) + eps)

def eval_dataset(gt_dir, pred_dir, gt_labels_json=None, pred_labels_json=None, bin_thr=0.5, save_csv="metrics.csv"):
    # Load labels if provided
    def _load_labels(p):
        if p and os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    gt_labels = _load_labels(gt_labels_json)
    pred_labels = _load_labels(pred_labels_json)

    img_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(pred_dir, "*.png"))])
    sm_obj, em_obj, wfm_obj, fm_obj, mae_obj = SAdapter(), EAdapter(), WFMAdapter(), FAdapter(), MAEAdapter()
    rows = []
    # c-metrics accumulators
    c_counts = 0
    cSm_sum = cEm_sum = cFw_sum = cF_sum = cIoU_sum = 0.0
    cMAE_sum = 0.0
    iou_list = []

    for img_id in tqdm(img_ids, ncols=80):
        gt_path = os.path.join(gt_dir, img_id + ".png")
        pred_path = os.path.join(pred_dir, img_id + ".png")
        if not os.path.isfile(gt_path) or not os.path.isfile(pred_path): continue
        gt = read_gray01(gt_path)
        pr = read_gray01(pred_path)
        pr = resize_to(pr, gt.shape)
        gt_bin = to_binary01(gt, 0.5)
        sm_obj.update(pr, gt_bin)
        em_obj.update(pr, gt_bin)
        wfm_obj.update(pr, gt_bin)
        fm_obj.update(pr, gt_bin)
        mae_obj.update(pr, gt_bin)
        # Per-image summaries using adapters for compatibility (best-effort)
        # S per-image
        try:
            s_tmp = SAdapter(); s_tmp.update(pr, gt_bin); s_val = float(s_tmp.compute())
        except Exception:
            s_val = None
        em_adp = None
        try:
            em_tmp = EAdapter(); em_tmp.update(pr, gt_bin); em_res = em_tmp.compute()
            if isinstance(em_res, dict):
                # Try common keys
                em_adp = em_res.get('adp', None)
                if em_adp is None and 'e' in em_res and isinstance(em_res['e'], dict):
                    em_adp = em_res['e'].get('adp', None)
        except Exception:
            em_adp = None
        fm_max = None
        try:
            fm_tmp = FAdapter(); fm_tmp.update(pr, gt_bin); fm_res = fm_tmp.compute()
            if isinstance(fm_res, dict):
                if 'fm' in fm_res and isinstance(fm_res['fm'], dict):
                    fm_max = fm_res['fm'].get('max', None)
                else:
                    fm_max = fm_res.get('max', None)
        except Exception:
            fm_max = None
        wfm_val = None
        try:
            wfm_tmp = WFMAdapter(); wfm_tmp.update(pr, gt_bin); wfm_res = wfm_tmp.compute()
            if isinstance(wfm_res, dict):
                # look for a scalar value under typical keys
                for k in ('wfm', 'WeightedF', 'value'):
                    if k in wfm_res and isinstance(wfm_res[k], (int, float)):
                        wfm_val = float(wfm_res[k])
                        break
            elif isinstance(wfm_res, (int, float)):
                wfm_val = float(wfm_res)
        except Exception:
            wfm_val = None
        mae_val = None
        try:
            mae_tmp = MAEAdapter(); mae_tmp.update(pr, gt_bin); mae_res = mae_tmp.compute()
            if isinstance(mae_res, (int, float)):
                mae_val = float(mae_res)
            elif isinstance(mae_res, dict):
                mae_val = float(list(mae_res.values())[0]) if mae_res else None
        except Exception:
            mae_val = None
        iou_val = iou_bin(to_binary01(pr, bin_thr), gt_bin)
        iou_list.append(iou_val)
        rows.append(dict(id=img_id, S=s_val, E_adp=em_adp, Fw_beta=wfm_val, Fbeta_max=fm_max, MAE=mae_val, IoU=iou_val))

        # Class-aware metrics if labels provided
        if gt_labels is not None and pred_labels is not None:
            gt_cls = gt_labels.get(img_id)
            pr_cls = pred_labels.get(img_id)
            if gt_cls is not None and pr_cls is not None:
                c_counts += 1
                if gt_cls == pr_cls:
                    cSm_sum += (s_val or 0.0)
                    cEm_sum += (em_adp or 0.0)
                    cFw_sum += (wfm_val or 0.0)
                    cF_sum  += (fm_max or 0.0)
                    cIoU_sum += (iou_val or 0.0)
                    cMAE_sum += (mae_val or 0.0)
                else:
                    cSm_sum += 0.0
                    cEm_sum += 0.0
                    cFw_sum += 0.0
                    cF_sum  += 0.0
                    cIoU_sum += 0.0
                    cMAE_sum += 1.0
    # Convert potential numpy types to pure python for JSON
    def _to_py(o):
        try:
            import numpy as _np
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, (dict, list, tuple)):
                if isinstance(o, dict):
                    return {k: _to_py(v) for k, v in o.items()}
                else:
                    return [ _to_py(v) for v in o ]
            if hasattr(o, 'item'):
                try:
                    return o.item()
                except Exception:
                    return o
            return o
        except Exception:
            return o

    cos_res = {
        "S_measure": sm_obj.compute(),
        "E_measure": em_obj.compute(),
        "F_measure": fm_obj.compute(),
        "Weighted_Fbeta": wfm_obj.compute(),
        "MAE": mae_obj.compute(),
        "IoU_mean": float(np.mean(iou_list)) if len(iou_list)>0 else None
    }
    # Class-aware results
    c_res = None
    if c_counts > 0:
        c_res = {
            "cSm": cSm_sum / c_counts,
            "cEm": cEm_sum / c_counts,
            "cFwβ": cFw_sum / c_counts,
            "cFβ":  cF_sum / c_counts,
            "cIoU": cIoU_sum / c_counts,
            "cMAE": cMAE_sum / c_counts,
            "count": c_counts
        }
    if rows: pd.DataFrame(rows).to_csv(save_csv, index=False)
    py_cos = _to_py(cos_res)
    py_c   = _to_py(c_res) if c_res is not None else None
    print({"COS": py_cos, "C": py_c})
    with open("summary.json", "w", encoding="utf-8") as f: json.dump({"COS": py_cos, "C": py_c}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_labels", default=None)
    ap.add_argument("--pred_labels", default=None)
    ap.add_argument("--out_csv", default="per_image_metrics.csv")
    ap.add_argument("--bin_thr", type=float, default=0.5)
    a = ap.parse_args()
    eval_dataset(a.gt_dir, a.pred_dir, a.gt_labels, a.pred_labels, a.bin_thr, a.out_csv)
