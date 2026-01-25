#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from PIL import Image, ImageDraw

# =========================
# 전역 설정
# =========================
#IMAGE_FOLDER = '/MIR/nas1/dreamyou070/data/HQ-Edit/export/input_image'
base_folder = '/MIR/nas1/dreamyou070/data/EmuEdit/export/test'
IMAGE_FOLDER = os.path.join(base_folder, 'input_image')
IMG_ZERO_ROOT = '/MIR/nas1/dreamyou070/Project/Project_EDIF/Zeroing/Zeroing_Experiment_Qwen2509/ImgZeroing_Non2509_AttentionControl'
TXT_ZERO_ROOT = '/MIR/nas1/dreamyou070/Project/Project_EDIF/Zeroing/Zeroing_Experiment_Qwen2509/ImgZeroing_Non2509_AttentionControl'
PAIRED_OUT_ROOT = '/MIR/nas1/dreamyou070/Project/Project_EDIF/Zeroing/Zeroing_Experiment_Qwen2509/ImgZeroing_Non2509_AttentionControl_concat2'
# 그리드 파일명
GRID_NAME_IMG = 'grid_image_zeroing_8x8.png'
GRID_NAME_TXT = 'grid_text_zeroing_8x8.png'

# 그리드/페어 이미지를 기존 것과 관계없이 새 규격으로 다시 만들고 싶으면 True
FORCE_REBUILD = True

# =========================
# 유틸 (그리드/결합/세이브)
# =========================
def make_grid(
    image_paths,
    grid_size=(8, 8),       # 8x8
    thumb_size=(256, 256),
    bg=(255, 255, 255),
    pad=5,                  # 셀 사이 간격
    inner_pad=4,            # 썸네일 내부 여백
    inner_bg=(245, 245, 245),
    draw_border=True,
    border_px=1,
    border_color=(220, 220, 220),
):
    """image_paths를 8x8 그리드로 배치하여 PIL 이미지 반환."""
    rows, cols = grid_size
    W, H = thumb_size

    canvas_w = cols * W + (cols - 1) * pad
    canvas_h = rows * H + (rows - 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    inner_w = max(1, W - 2 * inner_pad)
    inner_h = max(1, H - 2 * inner_pad)

    total_cells = rows * cols
    # 정확히 64칸 맞추기 (부족하면 빈칸, 많으면 자름)
    if len(image_paths) < total_cells:
        image_paths = image_paths + [None] * (total_cells - len(image_paths))
    else:
        image_paths = image_paths[:total_cells]

    for idx in range(total_cells):
        r, c = divmod(idx, cols)
        cell_x = c * (W + pad)
        cell_y = r * (H + pad)

        # 셀 배경
        cell = Image.new("RGB", (W, H), inner_bg)

        p = image_paths[idx]
        if p and os.path.exists(p):
            try:
                im = Image.open(p).convert("RGB")
                im = im.resize((inner_w, inner_h), Image.Resampling.BICUBIC)
            except Exception:
                im = Image.new("RGB", (inner_w, inner_h), (220, 220, 220))
        else:
            im = Image.new("RGB", (inner_w, inner_h), (240, 240, 240))

        # 셀 내부 여백 적용
        cell.paste(im, (inner_pad, inner_pad))

        # 선택: 테두리
        if draw_border and border_px > 0:
            d = ImageDraw.Draw(cell)
            for k in range(border_px):
                d.rectangle([k, k, W-1-k, H-1-k], outline=border_color)

        canvas.paste(cell, (cell_x, cell_y))
    return canvas

def concat_side_by_side(left_img_path, right_img_path, bg=(255, 255, 255), gap=32):
    """좌우 합치기."""
    L = Image.open(left_img_path).convert("RGB")
    R = Image.open(right_img_path).convert("RGB")
    H = max(L.height, R.height)
    canvas = Image.new("RGB", (L.width + gap + R.width, H), bg)
    canvas.paste(L, (0, 0))
    canvas.paste(R, (L.width + gap, 0))
    return canvas

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# =========================
# 헬퍼: 파일 정렬/수집
# =========================
_num_re = re.compile(r'_(\d+)_zeroing\.png$')

def sorted_zero_imgs_strict(folder, prefix, max_idx=59):
    """
    prefix='image' 또는 'text'
    0..max_idx의 '{prefix}_{i}_zeroing.png'를 정확히 그 순서로 수집.
    """
    out = []
    for i in range(0, max_idx + 1):
        cand = os.path.join(folder, f"{prefix}_{i}_zeroing.png")
        if os.path.exists(cand):
            out.append(cand)
    return out

def find_original_candidate(folder, prefix, src_image_path):
    """
    첫 칸(source) 후보:
      - 폴더 내 관례 파일명 우선
      - 없으면 CSV 기준 원본 이미지 사용
    """
    candidates = [
        'source.png', 'original.png', 'base.png',
        'org.png', 'src.png',
        f'{prefix}_source.png', f'{prefix}_original.png'
    ]
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            return p
    if src_image_path and os.path.exists(src_image_path):
        return src_image_path
    return None

def find_non_zeroing_candidate(folder, prefix):
    """두 번째 칸(non_zeroing) 후보."""
    candidates = [
        f'{prefix}_non_zeroing.png',
        'non_zeroing.png',
        'nonzero.png',
        'non_zero.png'
    ]
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            return p
    return None

def ensure_grid_from_existing(folder, prefix, grid_filename, src_image_path, force_rebuild=False):
    """
    8x8(64칸) 그리드 생성:
      [0] source
      [1] non_zeroing
      [2..63] prefix_0_zeroing.png ~ prefix_59_zeroing.png
    부족하면 빈칸으로 패딩, 많으면 64개로 자름.
    """
    grid_path = os.path.join(folder, grid_filename)
    if os.path.exists(grid_path) and not force_rebuild:
        return grid_path

    # 순서대로 수집
    src_path = find_original_candidate(folder, prefix, src_image_path)
    nonzero_path = find_non_zeroing_candidate(folder, prefix)
    zeros = sorted_zero_imgs_strict(folder, prefix, max_idx=59)  # 0..59

    ordered = []
    ordered.append(src_path if src_path else None)         # [0]
    ordered.append(nonzero_path if nonzero_path else None) # [1]
    ordered.extend(zeros)                                  # [2..]

    grid_img = make_grid(
        ordered,
        grid_size=(7, 10),          # 8x8 62
        thumb_size=(256, 256),
        pad=5,
        inner_pad=4,
        inner_bg=(245, 245, 245),
        draw_border=True,
        border_px=1,
        border_color=(220, 220, 220),
    )
    save_image(grid_img, grid_path)
    return grid_path

# =========================
# 메인
# =========================
def main():

    os.makedirs(PAIRED_OUT_ROOT, exist_ok=True)

    datas = os.listdir(IMAGE_FOLDER)

    for fname in datas :
        name, _ = os.path.splitext(fname)
        img_dir = os.path.join(IMG_ZERO_ROOT, name)
        txt_dir = os.path.join(TXT_ZERO_ROOT, name)

        print(f'img_dir : {img_dir}')
        if not os.path.isdir(img_dir) and not os.path.isdir(txt_dir):
            continue

        src_image_path = os.path.join(IMAGE_FOLDER, fname)
        img_grid = txt_grid = None
        if os.path.isdir(img_dir):
            print(f'making {img_dir}')
            img_grid = ensure_grid_from_existing(
                folder=img_dir,
                prefix='image',
                grid_filename=GRID_NAME_IMG,
                src_image_path=src_image_path,
                force_rebuild=FORCE_REBUILD,
            )
        if os.path.isdir(txt_dir):
            print(f'making {txt_dir}')
            txt_grid = ensure_grid_from_existing(
                folder=txt_dir,
                prefix='text',
                grid_filename=GRID_NAME_TXT,
                src_image_path=src_image_path,
                force_rebuild=FORCE_REBUILD,
            )

        # 좌우 합치기 (간격 조금 띄움)
        if img_grid and txt_grid and os.path.exists(img_grid) and os.path.exists(txt_grid):
            out_path = os.path.join(PAIRED_OUT_ROOT, f'{name}_image_text_grids.png')
            if not os.path.exists(out_path) or FORCE_REBUILD:
                merged = concat_side_by_side(img_grid, txt_grid, gap=32)
                save_image(merged, out_path)
            print(f"[INFO] paired: {out_path}")

if __name__ == '__main__':
    main()
