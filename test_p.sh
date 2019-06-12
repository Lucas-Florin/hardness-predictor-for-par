python train_baseline.py \
    -d pa100k \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --load-weights=2019-06-06_13-42-04_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib