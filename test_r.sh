python train_baseline.py \
    -d rap \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-06-03_17-42-43_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib