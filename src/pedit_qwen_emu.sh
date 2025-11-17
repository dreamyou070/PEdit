#!/usr/bin/env bash
GPU="0"                # CUDA_VISIBLE_DEVICES
START_IDX=0            # inclusive
END_IDX=300            # exclusive (like Python range)

export CUDA_VISIBLE_DEVICES="$GPU"

for (( i=START_IDX; i<END_IDX; i++ )); do
  python qwen_edit_optimize_hq.py config/qwen_emu_3_1.yml --target "$i"
done
