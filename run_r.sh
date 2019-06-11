python train_baseline.py \
    -d rap \
    --gpu-devices 0 \
    --max-epoch 120 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 20 40 \
    --loss-func deepmar \
    --use-macc
