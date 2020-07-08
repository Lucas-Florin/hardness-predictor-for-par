taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --hp-model resnet50_nh_strong \
    --gpu-devices 4 \
    --max-epoch 15 \
    --stepsize 14 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --optim-group-pretrained \
    --no-hp-feedback \
    --ap-baseline=../rp_rap/_checkpoint.pth.tar \
    --use-deepmar-for-hp \
    --fix-seed \
    --hp-net-lr-multiplier 1.0 \
    --train-batch-size 32