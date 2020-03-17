python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2020-03-05_15-22-31_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib \
    --f1-calib-split val \