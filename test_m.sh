python train_baseline.py \
    -d market1501 \
    --gpu-devices 1 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --load-weights=2019-05-28_16-50-43_checkpoint.pth.tar \
    --evaluate \
    --group-atts