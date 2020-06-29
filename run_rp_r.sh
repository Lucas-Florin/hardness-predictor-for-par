taskset -c 20-25 \
python realistic_predictor_trainer.py \
    -d rap \
    --hp-model resnet50_nh_strong \
    --gpu-devices 4 \
    --max-epoch 30 \
    --stepsize 16 22 27 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --ap-baseline=../rp_rap/_checkpoint.pth.tar \
    --use-deepmar-for-hp \
    --fix-seed \
    --hp-net-lr-multiplier 1.0