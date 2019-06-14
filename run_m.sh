python train_baseline.py \
    -d market1501 \
    --gpu-devices 1 \
    --max-epoch 3 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --stepsize 20 40 \
    --group-atts \
    --loss-func scel \
    --f1-calib
