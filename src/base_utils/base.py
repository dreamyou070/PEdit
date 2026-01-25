import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

def save_config(unique_folder, args):
    import os, json, datetime, pprint
    from dataclasses import asdict, is_dataclass

    config_dir = os.path.join(unique_folder, 'config.txt')

    def _to_plain(obj):
        # argparse.Namespace, dataclass, 기타를 dict로 정리
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_to_plain(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        return obj

    # 현재의 argument 를 config_dir 에 저장하기
    config_txt = os.path.join(unique_folder, "config.txt")
    config_json = os.path.join(unique_folder, "config.json")
    os.makedirs(unique_folder, exist_ok=True)
    plain_args = _to_plain(args)  # args: argparse.Namespace 또는 dataclass 등

    # 1) 사람이 읽기 쉬운 txt
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"# saved: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(pprint.pformat(plain_args, width=120, sort_dicts=False))

    # 2) 기계가 읽기 쉬운 json
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(plain_args, f, ensure_ascii=False, indent=2)


def load_image_bhwc_uint8(path: str, device: str = "cuda") -> Image.Image:
    img = Image.open(path).convert("RGB").resize((1024, 1024))
    return img
