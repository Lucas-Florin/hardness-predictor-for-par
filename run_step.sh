python train_baseline.py \
    -d pa100k \
    --gpu-devices 1 \
    --max-epoch 120 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --stepsize 40 80 \
    --print-freq 100 \
    --use-macc \
    --eval-freq 30