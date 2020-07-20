taskset -c 10-14 \
python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --load-weights=2020-07-08_20-03-59_checkpoint.pth.tar \
    --evaluate
    --no-cache
