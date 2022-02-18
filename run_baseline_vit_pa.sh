taskset -c 0-4 \
python baseline_trainer.py \
    -d pa100k \
    --height 256 \
    --width 256 \
    --model vit \
    --train-batch-size 8 \
    --gpu-devices 0 \
    --max-epoch 30 \
    --eval-freq 5 \
    --load-weights="./experiments/pretrained_vit-l16/imagenet21k+imagenet2012_ViT-L_16.pth" \
    --save-experiment="./experiments/transformer_baseline_pa/" \
    --optim-group-pretrained \
    --experiment-name="" \
    --weight-decay 0.0 \
    --lr 0.001 \
    --lr-scheduler cosine \

