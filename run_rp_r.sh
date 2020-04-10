taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --hp-model resnet50_nh \
    --gpu-devices 4 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --train-batch-size 64 \
    --ap-baseline=2020-03-28_16-40-49_checkpoint.pth.tar \
    --use-deepmar-for-hp
    --main-net-train-epochs 0 \
    --load-weights=../baseline_rap/2020-03-27_21-35-57_checkpoint.pth.tar
