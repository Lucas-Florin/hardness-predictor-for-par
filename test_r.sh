taskset -c 10-14 \
python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --load-weights=2020-07-03_15-20-07_checkpoint.pth.tar \
    --evaluate
    --no-cache