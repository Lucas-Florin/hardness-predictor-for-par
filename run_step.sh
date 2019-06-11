python train_baseline.py \
    --root datasets \
    -d peta \
    --gpu-devices 1 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --loss-func deepmar \
    --loss-func-param 0.85 \
    --use-macc \
    --group-atts

python train_baseline.py \
    --root datasets \
    -d peta \
    --gpu-devices 1 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --loss-func deepmar \
    --loss-func-param 0.70 \
    --use-macc \
    --group-atts