python baseline_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-06-04_13-26-12_checkpoint.pth.tar \
    --evaluate \
    --use-macc
    --f1-calib