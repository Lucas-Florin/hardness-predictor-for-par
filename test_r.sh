python baseline_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --load-weights=2020-03-27_21-35-57_best_checkpoint.pth.tar \
    --evaluate