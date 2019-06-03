python train_baseline.py \
    -d pa100k \
    --gpu-devices 0 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --load-weights=2019-05-31_02-36-24_checkpoint.pth.tar \
    --evaluate \
    --use-macc