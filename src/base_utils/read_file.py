import csv
from typing import List, Tuple, Dict
import numpy as np
import os

def load_csv_as_dicts(csv_path: str):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'fname': (r.get('fname') or '').strip(),
                'instruction': (r.get('instruction') or '').strip(),
                'task': (r.get('task') or '').strip(),
                'input_caption': (r.get('input_caption') or '').strip(),
                'output_caption': (r.get('output_caption') or '').strip(),
                'hash': (r.get('hash') or '').strip(),
                'idx': (r.get('idx') or '').strip(),})
    return rows



def _write_step_csv_per_case(
        step_idx: int,
        log_root: str,
        names: List[str],
        snr_raw: np.ndarray,
        vlm_raw: np.ndarray,
        kept_names: List[str],
        save_prompt: str,
        src_rec: Tuple[bool, float, float, str] = None,
        norm_ref= None,  # ← 추가
):

    def _minmax_norm_arr(arr: np.ndarray,
                         mn_mx= None,
                         eps: float = 1e-12,
                         clamp01: bool = True):
        """arr을 [0,1] 정규화. mn_mx 주어지면 그 범위를 그대로 사용(고정 기준)."""
        if arr.size == 0:
            return arr.astype(float), (0.0, 1.0)
        if mn_mx is None:
            a_min = float(np.nanmin(arr))
            a_max = float(np.nanmax(arr))
        else:
            a_min, a_max = mn_mx
        span = max(a_max - a_min, eps)
        out = (arr - a_min) / span
        if clamp01:
            out = np.clip(out, 0.0, 1.0)
        return out, (a_min, a_max)

    def _l2_to_one(x: np.ndarray, y: np.ndarray):
        """
        (0,0)->(x,y) 와 (0,0)->(1,1) 사이의 각도(라디안 또는 도)를 반환.
        각도는 [0, π] (degrees=True면 [0, 180]) 범위.
        x, y 는 동일 길이의 1D 배열(또는 브로드캐스팅 가능한 모양) 가정.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # 기준 대각선 단위벡터: (1,1)/||(1,1)|| = (1/√2, 1/√2)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)

        # (x,y) 벡터의 노름
        norm = np.sqrt(x * x + y * y)

        # (0,0)인 경우 각도가 정의되지 않음 → 각도 최댓값으로 치환(가장 불리하게)
        # 원하면 0으로 두고 별도 마스크로 처리해도 됨.
        zero_mask = (norm == 0)
        safe_norm = np.where(zero_mask, 1.0, norm)

        # cosθ = (v·d) / (||v||·||d||) = (x+y) / (norm * √2)
        cos_theta = (x + y) / (safe_norm * np.sqrt(2.0))

        # 수치 안정화
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)  # 라디안
        theta[zero_mask] = np.pi  # (0,0)은 최댓값으로 취급

        return theta

    snr_norm, _ = _minmax_norm_arr(snr_raw, norm_ref["snr"], clamp01=True)
    vlm_norm, _ = _minmax_norm_arr(vlm_raw, norm_ref["vlm"], clamp01=True)

    unique_folder = os.path.join(log_root, save_prompt)
    os.makedirs(unique_folder, exist_ok=True)
    kept_set = set(kept_names)
    d = _l2_to_one(snr_norm, vlm_norm)

    for idx, case_name in enumerate(names):
        # {case_name}
        csv_path = os.path.join(unique_folder, f"record.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["step", "case_name", "snr_raw", "vlm_raw", "snr_norm", "vlm_norm", "d_to_(1,1)", "kept"])
            w.writerow([
                int(step_idx),
                str(case_name),
                float(snr_raw[idx]),  # 원시 값 (음수 가능)
                float(vlm_raw[idx]),
                float(snr_norm[idx]),  # 0~1
                float(vlm_norm[idx]),  # 0~1
                float(d[idx]),
                int(case_name in kept_set),
            ])

    if src_rec is not None and src_rec[0]:
        _, src_snr, src_vlm, src_name = src_rec
        if norm_ref is not None:
            src_snr_norm, _ = _minmax_norm_arr(np.array([src_snr], dtype=float), norm_ref.get("snr"),
                                               clamp01=True)
            src_vlm_norm, _ = _minmax_norm_arr(np.array([src_vlm], dtype=float), norm_ref.get("vlm"),
                                               clamp01=True)
            src_d = float(_l2_to_one(src_snr_norm, src_vlm_norm)[0])
            snr_norm_val = float(src_snr_norm[0])
            vlm_norm_val = float(src_vlm_norm[0])
        else:
            src_d = 0.0
            snr_norm_val = ""
            vlm_norm_val = ""

        csv_path = os.path.join(unique_folder, f"{src_name}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["step", "snr_raw", "vlm_raw", "snr_norm", "vlm_norm", "d_to_(1,1)", "kept"])
            w.writerow([
                int(step_idx),
                float(src_snr),
                float(src_vlm),
                snr_norm_val,
                vlm_norm_val,
                src_d,
                0,  # kept=0 고정
            ])