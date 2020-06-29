taskset -c 10-14 \
python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --max-epoch 30 \
    --stepsize 16 22 27 \
    --eval-split test \
    --save-experiment=./storage/baseline_rap/ \
    --loss-func deepmar \
    --optim-group-pretrained \
    --fix-seed
