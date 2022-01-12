taskset -c 0-4 \
python baseline_trainer.py \
    -d pa100k \
    --height 224 \
    --width 224 \
    --model pvt_small \
    --train-batch-size 32 \
    --gpu-devices 0 \
    --max-epoch 30 \
    --eval-freq 5 \
    --save-experiment="./experiments/pvt_baseline_pa/" \
    --optim adamw \
    --weight-decay 0.05 \
    --lr 0.001 \
