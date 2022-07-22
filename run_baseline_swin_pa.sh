taskset -c 10-14 \
python baseline_trainer.py \
    -d pa100k \
    --height 224 \
    --width 224 \
    --model swin_t \
    --train-batch-size 64 \
    --gpu-devices 2 \
    --max-epoch 10 \
    --eval-freq 1 \
    --save-experiment="./experiments/swin_baseline_pa/" \
    --load-weights="./experiments/pretrained_swin/swin_tiny_patch4_window7_224.pth" \
    --optim-group-pretrained \
    --optim adamw \
    --weight-decay 0.05 \
    --lr 0.001 \
    --lr-scheduler cosine \
    --experiment-name="_"
