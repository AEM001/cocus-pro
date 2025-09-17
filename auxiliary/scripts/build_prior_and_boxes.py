"""
【LLM输出到SAM输入转换器】
作用：将LLM生成的行/列ID转换为SAM可用的初始掩码和边界框
核心功能：
  - 行/列ID解析：从LLM输出JSON中提取行列标识符
  - 掩码生成：基于行列ID的交集生成初始二值掩码
  - 边界框计算：从掩码计算最小外接矩形作为SAM输入
  - 数据验证：处理越界ID、空结果等异常情况

与上下游模块关系：
  - 接收LLM输出：从llm_out/{name}_output.json读取行列ID
  - 接收网格元数据：从out/{name}/{name}_meta.json读取行列边界
  - 输出SAM输入：生成box_out/{name}/目录下的prior_mask.png和sam_boxes.json
  - 供run_sculpt_simple.py和run_sculpt.py作为输入

技术特点：
  - 智能方向检测：自动识别行列ID是否被LLM交换
  - 容错处理：越界ID自动裁剪，空结果优雅处理
  - 标准化输出：统一目录结构和文件命名规范
  - 扩展支持：支持多种ID字段名格式兼容

工作流程：
  1. 加载LLM输出和网格元数据
  2. 解析并验证行列ID
  3. 生成初始掩码（行列交集并集）
  4. 计算边界框并可选扩展
  5. 输出标准化SAM输入文件
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于行/列 ids（来自 LLM 输出）构建 prior 掩码与 SAM 框。

输入约定：
- 当使用 --config config.yaml --name <name> 时，脚本会从 YAML 的 paths 中取：
  image, meta, pred，并将 {name} 替换为指定名称。
- 如果未提供 --config，但提供了 --name，本脚本也会按默认模板自动推断：
  image = images/{name}.png（如需 .jpg 可在命令行或 YAML 覆盖）
  meta  = out/{name}/{name}_meta.json
  pred  = llm_out/{name}_output.json  ← 按你的要求从 llm_out 读取 {name}_output.json

命令示例：
  python build_prior_and_boxes.py --config config.yaml --name q
  python build_prior_and_boxes.py --name q  # 无 YAML 也可，按默认模板推断路径
"""

import os, json, argparse
from PIL import Image
import numpy as np
from PIL import Image
import cv2

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def save_mask_png(mask01, out_path: str):
    """保存二值掩码为黑白PNG（0/255）。"""
    Image.fromarray((mask01.astype('uint8') * 255)).save(out_path)

def norm_box(box, W, H):
    """把 -1 处理成边界；并裁剪到 [0..W-1]/[0..H-1]，返回 (l,t,r,b)（含 r,b）"""
    l,t,r,b = box
    r = (W-1) if r == -1 else r
    b = (H-1) if b == -1 else b
    l,t,r,b = int(l), int(t), int(r), int(b)
    l, r = clamp(l,0,W-1), clamp(r,0,W-1)
    t, b = clamp(t,0,H-1), clamp(b,0,H-1)
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return l,t,r,b

def intersect_box(rb, cb, W, H):
    """行框(row box) ∩ 列框(col box)"""
    rl, rt, rr, rb_ = norm_box(rb, W, H)
    cl, ct, cr, cb_ = norm_box(cb, W, H)
    l, t = max(rl, cl), max(rt, ct)
    r, b = min(rr, cr), min(rb_, cb_)
    if r < l or b < t: return None
    return (l,t,r,b)

def ids_sanitize(ids, n_max, kind="horizontal"):
    """去重+排序+越界剪裁；返回合法 id 列表（1-based）"""
    if not ids: return []
    s = sorted(set([i for i in ids if 1 <= i <= n_max]))
    dropped = [i for i in ids if i < 1 or i > n_max]
    if dropped:
        print(f"[WARN] {kind} ids 越界被忽略: {dropped}，有效范围 1..{n_max}")
    return s

def count_invalid(ids, n_max):
    """统计越界个数（保留原始重复，不做去重）"""
    if not ids: return 0
    return sum(1 for i in ids if not (1 <= i <= n_max))

def to_int_list(v):
    """宽松地把 JSON 中的 ids 转成 int 列表（允许字符串数字）"""
    if v is None: return []
    if not isinstance(v, (list, tuple)): return []
    out = []
    for x in v:
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, str):
            x2 = x.strip()
            if x2.isdigit():
                out.append(int(x2))
            else:
                # 简单容错：诸如 "3." " 4 " 去掉尾巴再试
                try:
                    out.append(int(float(x2)))
                except Exception:
                    pass
    return out

def resolve_orientation(ids_v_raw, ids_h_raw, rows, cols, allow_swap=True):
    """
    根据数值范围自动判定 vertical / horizontal 是否被互换：
    - 方案 A：vertical→rows，horizontal→cols（与 make_region_prompts 约定一致）
    - 方案 B：SWAP：horizontal→rows，vertical→cols（若上游模型把语义反了）
    用越界总数与是否为空作为判据选择更合理的方案。
    返回：ids_v(final for rows), ids_h(final for cols), swapped(bool), reason(str)
    """
    # 统计越界数量（不去重）
    invA = count_invalid(ids_v_raw, rows) + count_invalid(ids_h_raw, cols)
    invB = count_invalid(ids_h_raw, rows) + count_invalid(ids_v_raw, cols)

    # 先各自 sanitize（去重+排序+越界剪）
    A_v = ids_sanitize(ids_v_raw, rows, "vertical→rows")
    A_h = ids_sanitize(ids_h_raw, cols, "horizontal→cols")
    B_v = ids_sanitize(ids_h_raw, rows, "SWAP horizontal→rows")
    B_h = ids_sanitize(ids_v_raw, cols, "SWAP vertical→cols")

    # 可用性判据
    scoreA = (invA, 0 if (A_v and A_h) else 1)  # 越界越少越好，且希望两侧都非空
    scoreB = (invB, 0 if (B_v and B_h) else 1)

    if not allow_swap:
        return ids_sanitize(ids_v_raw, rows, "vertical→rows"), ids_sanitize(ids_h_raw, cols, "horizontal→cols"), False, "按显式键语义映射（禁用自动交换）"
    if scoreB < scoreA:
        return B_v, B_h, True, f"选用 SWAP（invalid={invB} < {invA}）或更完整的非空组合"
    else:
        return A_v, A_h, False, f"选用默认映射（invalid={invA} <= {invB}）"

def draw_mask_from_cells(H, W, row_boxes, col_boxes, ids_v, ids_h):
    """对每个 (v,h) 单元格做并集，得到二值掩码"""
    M = np.zeros((H, W), np.uint8)
    for v in ids_v:
        for h in ids_h:
            rb = row_boxes[v-1]
            cb = col_boxes[h-1]
            cell = intersect_box(rb, cb, W, H)
            if cell is None: continue
            l,t,r,b = cell
            M[t:b+1, l:r+1] = 1
    return M

def dilate_mask(mask01, base_box, ratio=0.06, lo=3, hi=25):
    """像素级拓展：椭圆核膨胀。ratio 基于最短边，自动转成像素半径"""
    l,t,r,b = base_box
    bw, bh = (r-l+1), (b-t+1)
    rad = int(round(ratio * min(bw, bh)))
    rad = clamp(rad, lo, hi)
    if rad <= 0: return mask01
    k = 2*rad + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask01, kernel)

def bbox_from_mask(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def expand_box(box, W, H, scale):
    """以中心等比扩张，裁剪到图像范围"""
    l,t,r,b = box
    cx, cy = (l+r)/2, (t+b)/2
    bw, bh  = (r-l+1), (b-t+1)
    nw, nh  = bw*scale, bh*scale
    nl = int(round(cx - nw/2))
    nr = int(round(cx + nw/2))
    nt = int(round(cy - nh/2))
    nb = int(round(cy + nh/2))
    return (clamp(nl,0,W-1), clamp(nt,0,H-1), clamp(nr,0,W-1), clamp(nb,0,H-1))


def main():
    ap = argparse.ArgumentParser(description="根据LLM输出的行/列ids生成初始SAM boxes（推理-only）")
    ap.add_argument("--name", help="图片名，不含扩展名。如 q")
    ap.add_argument("--image", help="原图路径；若省略且提供 --name，则为 images/{name}.png")
    ap.add_argument("--meta", help="meta.json 路径；若省略且提供 --name，则为 out/{name}/{name}_meta.json")
    ap.add_argument("--pred", help="LLM输出json路径；若省略且提供 --name，则为 llm_out/{name}_output.json")
    ap.add_argument("--out",  default="../box_out", help="输出根目录，默认 ../box_out")
    args = ap.parse_args()

    # 推断相对路径
    if args.name:
        if not args.image:
            args.image = f"../../dataset/COD10K_TEST_DIR/Imgs/{args.name}.jpg"
        if not args.meta:
            args.meta = f"../out/{args.name}/{args.name}_meta.json"
        if not args.pred:
            args.pred = f"../llm_out/{args.name}_output.json"

    if not all([args.image, args.meta, args.pred]):
        raise ValueError("必须提供 image、meta、pred 路径，或提供 --name 以自动推断")

    meta = load_json(args.meta)
    pred = load_json(args.pred)

    base = os.path.splitext(os.path.basename(meta.get("image", args.image)))[0]
    # 将结果放到 box_out/{name}/（由 args.out 控制根目录）
    out_dir = os.path.join(args.out, base)
    os.makedirs(out_dir, exist_ok=True)

    W, H = int(meta["width"]), int(meta["height"])
    rows, cols = int(meta["rows"]), int(meta["cols"])
    row_boxes = meta["row_boxes"]
    col_boxes = meta["col_boxes"]

    # 宽松读取 + 自动判定横竖是否被互换
    # 兼容多种字段名：
    # - 旧: ids_vertical / ids_horizontal
    # - 新: ids_line_vertical / ids_line_horizontal（来自 llm_out/{name}_output.json）
    def get_first(pred_dict, keys, default=None):
        for k in keys:
            if k in pred_dict and pred_dict[k] is not None:
                return pred_dict[k], k
        return default, None

    v_val, v_key = get_first(pred, [
        "ids_line_vertical",  # 竖线（上→下），用于划分“列”（cols）
        "ids_vertical", "vertical_ids", "v_ids", "cols_ids"
    ], ([], None))
    h_val, h_key = get_first(pred, [
        "ids_line_horizontal",  # 横线（左→右），用于划分“行”（rows）
        "ids_horizontal", "horizontal_ids", "h_ids", "rows_ids"
    ], ([], None))

    # 注意：根据用户定义，“竖线”分列，“横线”分行
    ids_rows_raw = to_int_list(h_val)  # rows <- horizontal lines ids
    ids_cols_raw = to_int_list(v_val)  # cols <- vertical lines ids

    # 简单按语义直接映射（不做自动交换）：
    ids_v, ids_h, swapped, why = resolve_orientation(ids_rows_raw, ids_cols_raw, rows, cols)
    print(f"[INFO] {why}（横线→行, 竖线→列）")

    if not ids_v or not ids_h:
        print("[INFO] 空 ids，跳过。")
        return

    # 1) 交集单元并集 → 粗掩码
    M0 = draw_mask_from_cells(H, W, row_boxes, col_boxes, ids_v, ids_h)

    # 2) 从初始掩码得到基础 box
    B = bbox_from_mask(M0)
    if B is None:
        print("[WARN] 初始掩码为空，无法生成边界框。")
        return
    
    # 3) 直接使用基础 box 作为 SAM 的输入
    boxes = [B]

    # 保存结果：prior 掩码 + boxes
    mask_path = os.path.join(out_dir, f"{base}_prior_mask.png")
    save_mask_png(M0, mask_path)
    print("[OK] prior mask →", mask_path)

    boxes_path = os.path.join(out_dir, f"{base}_sam_boxes.json")
    with open(boxes_path, "w", encoding="utf-8") as f:
        json.dump({"boxes":[list(map(int,b)) for b in boxes]}, f, ensure_ascii=False, indent=2)
    print("[OK] boxes for SAM →", boxes_path)

    # 方便在终端快速查看
    print("Base box:", B)

if __name__ == "__main__":
    main()
