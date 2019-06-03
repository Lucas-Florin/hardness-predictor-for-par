python train_baseline.py \
    -d rap \
    --gpu-devices 0 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-05-30_00-22-19_checkpoint.pth.tar \
    --evaluate \
    --use-macc