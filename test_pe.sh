python baseline_trainer.py \
    -d peta \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --load-weights=2019-06-11_14-45-47_checkpoint.pth.tar \
    --f1-calib \
    --evaluate \
    --use-macc