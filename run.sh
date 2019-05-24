python train_baseline.py \
    --root datasets \
    -d market1501 \
    --gpu-devices 2 \
    --max-epoch 60 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --lr-scheduler single_step \
    --print-freq 100 \
