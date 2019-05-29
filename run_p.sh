python train_baseline.py \
    --root datasets \
    -d pa100k \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --stepsize 20 40 \
    --loss-func scel \
    --use-macc
