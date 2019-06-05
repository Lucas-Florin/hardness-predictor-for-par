python train_baseline.py \
    --root datasets \
    -d peta \
    --gpu-devices 3 \
    --max-epoch 3 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --stepsize 20 40 \
    --loss-func deepmar \
    --use-macc

python train_baseline.py \
    --root datasets \
    -d peta \
    --gpu-devices 3 \
    --max-epoch 3 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_peta/ \
    --stepsize 20 40 \
    --loss-func sscel \
    --use-macc