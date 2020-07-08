taskset -c 10-14 \
python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --max-epoch 30 \
    --stepsize 15 20 25 \
    --train-batch-size 32 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --loss-func deepmar \
    --optim-group-pretrained \
    --fix-seed

