#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_region_prompts.py
将输入图片按“行/列”划分，并生成两张带编号的提示图：
- *_vertical.png  ：按垂直方向划分（画横线），在每一“行”的中心标号 1..rows
- *_horizontal.png：按水平方向划分（画竖线），在每一“列”的中心标号 1..cols
同时导出一个 meta.json，记录每一行/列的像素边界，方便后续由 ids_vertical / ids_horizontal 计算 ROI。

两种用法：
1) 纯命令行参数：
   python make_region_prompts.py \
     --image images/cat.jpg \
     --rows 9 --cols 9 \
     --outdir out \
     --line-thickness 3 \
     --font-size 40 \
     --font /System/Library/Fonts/Supplemental/Arial.ttf

2) 从 YAML 统一加载：
   python make_region_prompts.py --config config.yaml --name cat
   - 其中 config.yaml 的 paths.image 使用 {name} 占位符，例如 images/{name}.jpg
   - make_region_prompts 节设置 rows/cols/line_thickness/font_scale/font_size/font 等默认值

若不提供 --font，将使用系统可用字体或 Pillow 的默认字体。
"""
import argparse
import json
import os
from typing import List, Tuple, Optional, Any

from PIL import Image, ImageDraw, ImageFont


def load_font(font_path: Optional[str], font_size: int) -> Any:
    """优先加载指定 TTF/ TTC；未提供时在 macOS 常见系统字体中选可用的。
    若均不可用，再退回 Pillow 的位图字体（不可缩放）。
    """
    # 1) 用户显式提供字体
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

    # 2) 尝试常见系统字体（macOS + 通用）
    candidates = [
        # macOS 常见路径
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        # Pillow 打包的常见字体（若存在）
        "DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                return ImageFont.truetype(p, font_size)
            # 某些环境下直接给出字体名也可解析
            if os.path.sep not in p:
                return ImageFont.truetype(p, font_size)
        except Exception:
            continue

    # 3) 兜底：默认位图字体（不支持缩放，可能看起来很小）
    return ImageFont.load_default()


def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(255, 255, 255),
    outline=(0, 0, 0),
    outline_width: int = 0,
):
    x, y = xy
    # 先画黑色描边
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    # 再画白色文字
    draw.text((x, y), text, font=font, fill=fill)


def make_vertical_prompt(
    img: Image.Image,
    rows: int,
    line_thickness: int,
    font: ImageFont.ImageFont,
    line_color=(0, 255, 0),
    text_color=(255, 0, 0),
) -> Image.Image:
    """
    垂直方向划分：画“横线”，把图像切成 rows 个“行”，并在每一行居中标注 1..rows
    """
    w, h = img.size
    y_step = h / rows
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)

    # 画横线（行分割线）
    for i in range(1, rows):  # 中间线（不含边界）
        y = int(round(i * y_step))
        draw.rectangle([(0, y - line_thickness // 2), (w, y + line_thickness // 2)], fill=line_color)

    # 标号：每一行中心 (w/2, (i-0.5)*y_step)
    for i in range(1, rows + 1):
        cy = int(round((i - 0.5) * y_step))
        text = str(i)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw_text_with_outline(draw, (w // 2 - tw // 2, cy - th // 2), text, font, fill=text_color)

    return canvas


def make_horizontal_prompt(
    img: Image.Image,
    cols: int,
    line_thickness: int,
    font: ImageFont.ImageFont,
    line_color=(0, 255, 0),
    text_color=(255, 0, 0),
) -> Image.Image:
    """
    水平方向划分：画“竖线”，把图像切成 cols 个“列”，并在每一列居中标注 1..cols
    """
    w, h = img.size
    x_step = w / cols
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)

    # 画竖线（列分割线）
    for j in range(1, cols):
        x = int(round(j * x_step))
        draw.rectangle([(x - line_thickness // 2, 0), (x + line_thickness // 2, h)], fill=line_color)

    # 标号：每一列中心 ((j-0.5)*x_step, h/2)
    for j in range(1, cols + 1):
        cx = int(round((j - 0.5) * x_step))
        text = str(j)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw_text_with_outline(draw, (cx - tw // 2, h // 2 - th // 2), text, font, fill=text_color)

    return canvas


def compute_row_boxes(h: int, rows: int) -> List[Tuple[int, int, int, int]]:
    """返回每一行的 (left, top, right, bottom)，行号 1..rows 对应索引 0..rows-1"""
    boxes = []
    y_step = h / rows
    for i in range(rows):
        top = int(round(i * y_step))
        bottom = int(round((i + 1) * y_step)) - 1
        boxes.append((0, max(0, top), -1, max(0, bottom)))  # right=-1 表示用图宽
    return boxes


def compute_col_boxes(w: int, cols: int) -> List[Tuple[int, int, int, int]]:
    """返回每一列的 (left, top, right, bottom)，列号 1..cols 对应索引 0..cols-1"""
    boxes = []
    x_step = w / cols
    for j in range(cols):
        left = int(round(j * x_step))
        right = int(round((j + 1) * x_step)) - 1
        boxes.append((max(0, left), 0, max(0, right), -1))  # bottom=-1 表示用图高
    return boxes


def ids_to_bbox(
    ids_vertical: List[int],
    ids_horizontal: List[int],
    w: int,
    h: int,
    rows: int,
    cols: int,
    pad_ratio_h: float = 0.2,
    pad_ratio_w: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    根据 1..rows 的 ids_vertical 与 1..cols 的 ids_horizontal 计算联合 ROI 的像素 bbox，
    并按比例添加 padding（与图像高/宽成比例）。
    返回 (left, top, right, bottom)，已裁剪到图像范围内。
    """
    if not ids_vertical or not ids_horizontal:
        return (0, 0, w - 1, h - 1)

    vy0 = min(ids_vertical) - 1
    vy1 = max(ids_vertical) - 1
    hx0 = min(ids_horizontal) - 1
    hx1 = max(ids_horizontal) - 1

    y_step = h / rows
    x_step = w / cols

    top = int(round(vy0 * y_step))
    bottom = int(round((vy1 + 1) * y_step)) - 1
    left = int(round(hx0 * x_step))
    right = int(round((hx1 + 1) * x_step)) - 1

    # padding
    pad_h = int(round((bottom - top + 1) * pad_ratio_h))
    pad_w = int(round((right - left + 1) * pad_ratio_w))

    top = max(0, top - pad_h)
    bottom = min(h - 1, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(w - 1, right + pad_w)

    return (left, top, right, bottom)


def main():
    parser = argparse.ArgumentParser(description="Generate region-aware visual prompts with numbered rows/cols.")
    # 简洁命令行参数（无需YAML）
    parser.add_argument("--image", help="输入图片路径，如 images/q.png。若未提供且指定 --name，则自动从 images/{name}.png 读取")
    parser.add_argument("--name", help="图片名（不含扩展名）。设置后可省略 --image")
    parser.add_argument("--rows", type=int, default=9, help="行数，默认9")
    parser.add_argument("--cols", type=int, default=9, help="列数，默认9")
    parser.add_argument("--outdir", default="./out", help="输出根目录，默认 ./out")
    parser.add_argument("--line-thickness", dest="line_thickness", type=int, default=3, help="分割线粗细（像素），默认3")
    parser.add_argument("--font-size", dest="font_size", type=int, help="标号字体大小；优先于 --font-scale")
    parser.add_argument("--font-scale", dest="font_scale", type=float, default=0.6, help="相对单元格最短边的比例(0~1)来确定字号，默认0.6")
    parser.add_argument("--font", help="可选：TTF 字体路径（例如 Arial/SimHei 等）")
    args = parser.parse_args()

    # 解析图片路径
    image_path = args.image
    if not image_path and args.name:
        image_path = os.path.join("./images", f"{args.name}.png")
    if not image_path:
        raise ValueError("必须提供 --image 或 --name（将从 images/{name}.png 推断）")

    merged = dict(
        image=image_path,
        rows=args.rows,
        cols=args.cols,
        outdir=args.outdir,
        line_thickness=args.line_thickness,
        font_size=args.font_size,
        font_scale=args.font_scale,
        font=args.font,
    )

    # 校验
    assert merged["image"], "必须提供 --image 或 --name"
    assert merged["rows"] is not None and merged["cols"] is not None, "必须提供 rows/cols"
    assert merged["rows"] >= 1 and merged["cols"] >= 1, "rows/cols 必须 >= 1"

    base = os.path.splitext(os.path.basename(merged["image"]))[0]
    out_dir = os.path.join(merged["outdir"], base)

    # 自动字体大小：若未设定，依据单元格尺寸与 font_scale 估计
    if merged["font_size"] is None:
        img_probe = Image.open(merged["image"])  # 延后复用
        w_probe, h_probe = img_probe.size
        cell_min = min(h_probe / float(merged["rows"]), w_probe / float(merged["cols"]))
        merged["font_size"] = max(12, int(round(cell_min * float(merged.get("font_scale", 0.6)))))
        img_probe.close()

    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(merged["image"]).convert("RGB")
    w, h = img.size

    font = load_font(merged["font"], merged["font_size"])

    # 生成两张提示图
    vertical_img = make_vertical_prompt(
        img,
        rows=int(merged["rows"]),
        line_thickness=int(merged["line_thickness"]),
        font=font,
    )
    horizontal_img = make_horizontal_prompt(
        img,
        cols=int(merged["cols"]),
        line_thickness=int(merged["line_thickness"]),
        font=font,
    )

    # 保存
    vertical_path = os.path.join(out_dir, f"{base}_vertical_{merged['rows']}.png")
    horizontal_path = os.path.join(out_dir, f"{base}_horizontal_{merged['cols']}.png")
    vertical_img.save(vertical_path)
    horizontal_img.save(horizontal_path)

    # 导出元信息（行/列像素边界，便于把 MLLM 输出的 ids 转成 bbox）
    meta = {
        "image": merged["image"],
        "width": w,
        "height": h,
        "rows": int(merged["rows"]),
        "cols": int(merged["cols"]),
        "row_boxes": compute_row_boxes(h, int(merged["rows"])),  # (left, top, right, bottom)，right/bottom 为 -1 表示到边界
        "col_boxes": compute_col_boxes(w, int(merged["cols"])),
        "note": "行号/列号均为 1 开始计数；与图片上的编号一致。",
    }
    meta_path = os.path.join(out_dir, f"{base}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print("  ", vertical_path)
    print("  ", horizontal_path)
    print("  ", meta_path)

    # 如果你已经拿到了 MLLM 的 ids_vertical / ids_horizontal，也可以在此测试一次 bbox 计算：
    # 示例（可删除）：
    # demo_bbox = ids_to_bbox(ids_vertical=[4,5,6], ids_horizontal=[5,6,7,8],
    #                         w=w, h=h, rows=args.rows, cols=args.cols,
    #                         pad_ratio_h=0.2, pad_ratio_w=0.2)
    # print("Demo bbox:", demo_bbox)


if __name__ == "__main__":
    main()
