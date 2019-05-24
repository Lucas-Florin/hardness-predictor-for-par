python train_baseline.py \
    --root datasets \
    -d market1501 \
    --gpu-devices 3 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --load-weights=2019-05-24_17-02-16_checkpoint.pth.tar \
    --evaluate \
    --group-atts \
    --use-macc