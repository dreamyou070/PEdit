#!/usr/bin/env bash
GPU="0"                # CUDA_VISIBLE_DEVICES
START_IDX=0            # 100 # 200
END_IDX=100            # 200 # 300

export CUDA_VISIBLE_DEVICES="$GPU"

for (( i=START_IDX; i<END_IDX; i++ )); do
  python pedit_qwen.py config/qwen_emu_5_1.yml --target "$i"
done
