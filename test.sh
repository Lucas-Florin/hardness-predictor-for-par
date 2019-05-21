python train_baseline.py \
    --root datasets \
    -d rap \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-05-20_14-22-52_checkpoint.pth.tar \
    --evaluate