python baseline_trainer.py \
    -d pa100k \
    --gpu-devices 0 \
    --max-epoch 30 \
    --save-experiment=./experiments/baseline_pa/ \
    --optim-group-pretrained \
    --fix-seed
