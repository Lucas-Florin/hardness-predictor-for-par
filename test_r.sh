taskset -c 0-4 \
python baseline_trainer.py \
    -d rap \
    -m resnet50_strong \
    --gpu-devices 0 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --load-weights=2020-06-26_15-09-29_imported_checkpoint.pth.tar \
    --evaluate
    --no-cache