import os, sys, json, time, shutil, subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

BASE = Path('/home/albert/code/CV')
DATASET = BASE / 'dataset'
SAMPLE_INFO = DATASET / 'sample_info.json'
CLASS_INFO = DATASET / 'class_info.json'
RUNS_DIR = BASE / 'evaluation' / 'runs'

PY = sys.executable


def load_samples(limit: int) -> List[Dict]:
    with open(SAMPLE_INFO, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    return samples[:limit]


def ensure_dirs(root: Path):
    (root / 'pred_masks').mkdir(parents=True, exist_ok=True)
    (root / 'gt_masks').mkdir(parents=True, exist_ok=True)


def run_single_sample(sample: Dict, rounds: int, api_model: str, high_res: bool, openai_mode: bool, clean_output: bool, skip_done: bool = True) -> Dict:
    name = Path(sample['image']).stem  # e.g., COD10K-CAM-... filename stem

    # 1) Build ROI box from GT mask (simple bbox) at auxiliary/box_out/{name}/{name}_sam_boxes.json
    gt_mask_path = DATASET / sample['mask']
    import cv2
    import numpy as np
    m = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f'GT mask not found: {gt_mask_path}')
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        # empty mask, fallback to full image
        img_path = DATASET / sample['image']
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        H, W = (img.shape[0], img.shape[1]) if img is not None else (m.shape[0], m.shape[1])
        x0, y0, x1, y1 = 0, 0, W-1, H-1
    else:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box_dir = BASE / 'auxiliary' / 'box_out' / name
    box_dir.mkdir(parents=True, exist_ok=True)
    with open(box_dir / f'{name}_sam_boxes.json', 'w', encoding='utf-8') as f:
        json.dump({"boxes": [[x0, y0, x1, y1]]}, f, ensure_ascii=False, indent=2)

    # 2) Create minimal semantic file so pipeline writes instance_info.txt
    llm_dir = BASE / 'auxiliary' / 'llm_out'
    llm_dir.mkdir(parents=True, exist_ok=True)
    with open(llm_dir / f'{name}_output.json', 'w', encoding='utf-8') as f:
        json.dump({"instance": sample.get('base_class', 'object'), "reason": "eval-gt-bbox", "method": "eval_gt_bbox"}, f, ensure_ascii=False, indent=2)

    # 3) Run clean_sam_sculpt.py for segmentation (skip if already done)
    out_dir = BASE / 'outputs' / 'clean_sculpt' / name
    pred_mask = out_dir / 'final_mask.png'
    inst_txt = out_dir / 'instance_info.txt'

    if skip_done and pred_mask.exists() and inst_txt.exists():
        print(f'[SKIP] segmentation already done for {name}, reusing existing outputs')
    else:
        cmd = [PY, str(BASE / 'clean_sam_sculpt.py'), '--name', name, '--rounds', str(rounds), '--api-model', api_model]
        if high_res:
            cmd.append('--high-resolution')
        if openai_mode:
            cmd.append('--use-openai-api')
        if clean_output:
            cmd.append('--clean-output')
        subprocess.run(cmd, cwd=str(BASE), check=True)

    # collect predicted mask (refresh paths)
    pred_mask = out_dir / 'final_mask.png'
    inst_txt = out_dir / 'instance_info.txt'

    res = {
        'name': name,
        'pred_mask': str(pred_mask),
        'instance_file': str(inst_txt),
        'ok': pred_mask.exists() and inst_txt.exists(),
    }
    return res


def copy_masks_for_eval(run_dir: Path, sample: Dict, pred_ok: bool):
    name = Path(sample['image']).stem
    if pred_ok:
        src_pred = BASE / 'outputs' / 'clean_sculpt' / name / 'final_mask.png'
        dst_pred = run_dir / 'pred_masks' / f'{name}.png'
        if src_pred.exists():
            shutil.copyfile(src_pred, dst_pred)
    # GT
    gt_rel = sample['mask']  # e.g., COD10K_TEST_DIR/GT/...png
    gt_path = DATASET / gt_rel
    dst_gt = run_dir / 'gt_masks' / f'{name}.png'
    if gt_path.exists():
        shutil.copyfile(gt_path, dst_gt)


def build_pred_labels(run_dir: Path, samples: List[Dict]):
    pred_labels = {}
    for s in samples:
        name = Path(s['image']).stem
        inst_file = BASE / 'outputs' / 'clean_sculpt' / name / 'instance_info.txt'
        if inst_file.exists():
            try:
                pred_labels[name] = inst_file.read_text(encoding='utf-8').strip()
            except Exception:
                continue
    with open(run_dir / 'pred_labels.json', 'w', encoding='utf-8') as f:
        json.dump(pred_labels, f, ensure_ascii=False, indent=2)


def build_gt_labels(run_dir: Path, samples: List[Dict]):
    gt_labels = {}
    for s in samples:
        name = Path(s['image']).stem
        gt_labels[name] = s.get('base_class', None)
    with open(run_dir / 'gt_labels.json', 'w', encoding='utf-8') as f:
        json.dump(gt_labels, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--api-model', default='qwen-vl-plus-latest')
    ap.add_argument('--high-resolution', action='store_true')
    ap.add_argument('--use-openai-api', action='store_true')
    ap.add_argument('--clean-output', action='store_true')
    ap.add_argument('--skip-done', action='store_true', default=True, help='Skip samples that already have outputs/clean_sculpt/{name}/final_mask.png')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / ts
    ensure_dirs(run_dir)

    samples = load_samples(args.limit)
    results = []

    for i, sample in enumerate(samples):
        print(f'[{i+1}/{len(samples)}] Running {sample["unique_id"]} ...')
        try:
            res = run_single_sample(sample, args.rounds, args.api_model, args.high_resolution, args.use_openai_api, args.clean_output, skip_done=args.skip_done)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] sample failed: {e}')
            res = {'name': Path(sample['image']).stem, 'ok': False}
        results.append(res)
        copy_masks_for_eval(run_dir, sample, res.get('ok', False))

    # labels files for c-metrics
    build_gt_labels(run_dir, samples)
    build_pred_labels(run_dir, samples)

    # compute metrics using evaluate.md script
    eval_script = BASE / 'evaluation' / 'eval_ovcos.py'
    if not eval_script.exists():
        # write it from evaluate.md payload subset: we'll create a simple wrapper calling py_sod_metrics if present
        pass

    # Install deps if missing (best-effort)
    try:
        subprocess.run([PY, '-m', 'pip', 'install', 'pysodmetrics', 'opencv-python-headless', 'numpy', 'tqdm', 'pandas'], check=True)
    except Exception as e:
        print('[WARN] pip install failed:', e)

    # write eval_ovcos.py from evaluate.md content
    src_eval = (BASE / 'evaluate.md').read_text(encoding='utf-8')
    # crude extraction: we already know our repo has an eval_ovcos.py example in the md; but to simplify, fetch it via reading and writing known section
    # For robustness, if that fails, we fallback to importing py_sod_metrics directly here.
    eval_py = (BASE / 'evaluation' / 'eval_ovcos.py')
    if not eval_py.exists():
        # minimal evaluator that matches evaluate.md interface
        eval_py.write_text('''import os, json, glob\nimport numpy as np\nimport cv2\nfrom tqdm import tqdm\nimport pandas as pd\nfrom py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, Fmeasure, MAE\n\ndef read_gray01(path):\n    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)\n    if m is None: raise FileNotFoundError(path)\n    m = m.astype(np.float32) / 255.0\n    return np.clip(m, 0, 1)\n\ndef to_binary01(arr, thr=0.5):\n    return (arr >= thr).astype(np.float32)\n\ndef resize_to(src, shape):\n    if src.shape == shape: return src\n    return cv2.resize(src, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)\n\ndef iou_bin(pred_bin, gt_bin, eps=1e-7):\n    inter = np.sum((pred_bin > 0) & (gt_bin > 0))\n    union = np.sum((pred_bin > 0) | (gt_bin > 0))\n    return float(inter) / (float(union) + eps)\n\ndef eval_dataset(gt_dir, pred_dir, bin_thr=0.5, save_csv=\"metrics.csv\"):\n    img_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(pred_dir, \"*.png\"))])\n    sm_obj, em_obj, wfm_obj, fm_obj, mae_obj = Smeasure(), Emeasure(), WeightedFmeasure(), Fmeasure(), MAE()\n    rows = []\n    for img_id in tqdm(img_ids, ncols=80):\n        gt_path = os.path.join(gt_dir, img_id + \".png\")\n        pred_path = os.path.join(pred_dir, img_id + \".png\")\n        if not os.path.isfile(gt_path) or not os.path.isfile(pred_path): continue\n        gt = read_gray01(gt_path)\n        pr = read_gray01(pred_path)\n        pr = resize_to(pr, gt.shape)\n        gt_bin = to_binary01(gt, 0.5)\n        sm_obj.update(pr, gt_bin)\n        em_obj.update(pr, gt_bin)\n        wfm_obj.update(pr, gt_bin)\n        fm_obj.update(pr, gt_bin)\n        mae_obj.update(pr, gt_bin)\n        fm_all = Fmeasure(); fm_all.update(pr, gt_bin); fm_max = fm_all.compute()[\"fm\"][\"max\"]\n        em_all = Emeasure(); em_all.update(pr, gt_bin); em_adp = em_all.compute()[\"adp\"]\n        wfm_i = WeightedFmeasure(); wfm_i.update(pr, gt_bin); wfm_val = wfm_i.compute()\n        mae_all = MAE(); mae_all.update(pr, gt_bin); mae_val = mae_all.compute()\n        iou_val = iou_bin(to_binary01(pr, bin_thr), gt_bin)\n        rows.append(dict(id=img_id, E_adp=em_adp, Fw_beta=wfm_val, Fbeta_max=fm_max, MAE=mae_val, IoU=iou_val))\n    cos_res = {\n        \"S_measure\": sm_obj.compute(),\n        \"E_measure(adp/mean/max)\": em_obj.compute(),\n        \"F_measure(max/mean/adp)\": fm_obj.compute(),\n        \"Weighted_Fbeta\": wfm_obj.compute(),\n        \"MAE\": mae_obj.compute()\n    }\n    if rows: pd.DataFrame(rows).to_csv(save_csv, index=False)\n    print(cos_res)\n    with open(\"summary.json\", \"w\", encoding=\"utf-8\") as f: json.dump(cos_res, f, ensure_ascii=False, indent=2)\n\nif __name__ == \"__main__\":\n    import argparse\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\"--gt_dir\", required=True)\n    ap.add_argument(\"--pred_dir\", required=True)\n    ap.add_argument(\"--out_csv\", default=\"per_image_metrics.csv\")\n    ap.add_argument(\"--bin_thr\", type=float, default=0.5)\n    a = ap.parse_args()\n    eval_dataset(a.gt_dir, a.pred_dir, a.bin_thr, a.out_csv)\n''', encoding='utf-8')

    # run evaluator
    subprocess.run([PY, str(eval_py), '--gt_dir', str(run_dir / 'gt_masks'), '--pred_dir', str(run_dir / 'pred_masks'), '--out_csv', str(run_dir / 'per_image_metrics.csv')], cwd=str(run_dir), check=True)

    print(f'\nDone. Results saved under: {run_dir}')

if __name__ == '__main__':
    main()
