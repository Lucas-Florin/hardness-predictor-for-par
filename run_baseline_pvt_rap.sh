taskset -c 0-4 \
python baseline_trainer.py \
    -d rap \
    --height 224 \
    --width 224 \
    --model pvt_small \
    --train-batch-size 64 \
    --gpu-devices 0 \
    --max-epoch 30 \
    --save-experiment="./experiments/pvt_baseline_rap/" \
    --load-weights="./experiments/pretrained_pvt/pvt_small.pth" \
    --optim-group-pretrained \
    --optim adamw \
    --weight-decay 0.05 \
    --lr 0.001 \
    --lr-scheduler cosine \
    --experiment-name=""
