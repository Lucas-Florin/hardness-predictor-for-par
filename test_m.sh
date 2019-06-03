python train_baseline.py \
    -d market1501 \
    --gpu-devices 2 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --load-weights=2019-05-30_08-13-38_checkpoint.pth.tar \
    --evaluate \
    --group-atts