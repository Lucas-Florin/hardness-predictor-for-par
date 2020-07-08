taskset -c 20-24 \
python baseline_trainer.py \
    -d pa100k \
    --gpu-devices 4 \
    --max-epoch 30 \
    --stepsize 15 20 25 \
    --train-batch-size 32 \
    --eval-split test \
    --save-experiment=./storage/baseline_pa/ \
    --loss-func deepmar \
    --optim-group-pretrained \
    --fix-seed
