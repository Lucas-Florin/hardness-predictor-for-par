python train_baseline.py \
    --root datasets \
    -d rap \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --lr-scheduler single_step \
    --print-freq 100
