python baseline_trainer.py \
    -d market1501 \
    --gpu-devices 0 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --load-weights=2019-06-03_16-37-07_checkpoint.pth.tar \
    --evaluate \
    --group-atts \
    --f1-calib