python train_baseline.py \
    --root datasets \
    -d peta \
    --gpu-devices 3 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --loss-func scel \
    --use-macc \
    --group-atts

