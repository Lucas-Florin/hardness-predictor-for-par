python baseline_trainer.py \
    --root datasets \
    -d peta \
    --gpu-devices 0 \
    --max-epoch 2 \
    --stepsize 20 40 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --loss-func scel \
    --f1-calib \
    --use-macc \
    --color-jitter

