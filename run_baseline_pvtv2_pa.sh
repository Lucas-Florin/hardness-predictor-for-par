taskset -c 20-24 \
python baseline_trainer.py \
    -d pa100k \
    --height 224 \
    --width 224 \
    --model pvt_v2_b2 \
    --train-batch-size 64 \
    --gpu-devices 4 \
    --max-epoch 5 \
    --eval-freq 1 \
    --save-experiment="./experiments/pvtv2_baseline_pa/" \
    --load-weights="./experiments/pretrained_pvtv2/pvt_v2_b2.pth" \
    --optim adamw \
    --weight-decay 0.5 \
    --lr 0.001 \
    --experiment-name="_"
