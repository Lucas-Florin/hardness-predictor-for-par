python train_baseline.py \
    --root datasets \
    -d market1501 \
    --gpu-devices 1 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --load-weights=2019-05-20_09-53-48_checkpoint.pth.tar \
    --evaluate