taskset -c 0-4 \
python baseline_trainer.py \
    -d rap \
    --height 224 \
    --width 224 \
    --model swin_t \
    --train-batch-size 64 \
    --gpu-devices 0 \
    --max-epoch 5 \
    --eval-freq 1 \
    --save-experiment="./experiments/swin_baseline_rap/" \
    --load-weights="./experiments/pretrained_swin/swin_tiny_patch4_window7_224.pth" \
    --optim-group-pretrained \
    --optim adamw \
    --weight-decay 0.05 \
    --lr 0.001 \
    --lr-scheduler cosine \
    --experiment-name="_"
