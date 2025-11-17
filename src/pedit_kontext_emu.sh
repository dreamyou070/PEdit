GPU="0"          # (필요하면 쓰고, CUDA_VISIBLE_DEVICES로 강제도 가능)
START_IDX=0
END_IDX=300
#CONFIG="config/kontext_emu_1_5.yml"
CONFIG="config/kontext_emu_1_3.yml"
#CONFIG="config/kontext_emu_1_1.yml"
#CONFIG="config/kontext_emu_3_1.yml"
#CONFIG="config/kontext_emu_5_1.yml"

export CUDA_VISIBLE_DEVICES=$GPU
echo "Running from $START_IDX to $END_IDX on GPU $CUDA_VISIBLE_DEVICES"
for (( i=$START_IDX; i<$END_IDX; i++ )); do
    python kontext_optimize.py "$CONFIG" --target "$i"
done