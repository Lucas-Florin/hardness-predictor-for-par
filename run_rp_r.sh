taskset -c 30-34 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 6 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/test/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --train-batch-size 64 \
    --ap-baseline=../rp_rap/2020-03-28_16-40-49_checkpoint.pth.tar \
    --use-deepmar-for-hp \
    --hp-calib linear \
    --f1-baseline=../baseline_rap/2020-03-27_21-35-57_checkpoint.pth.tar \
    --main-net-train-epochs 0 \
    --load-weights=../baseline_rap/2020-03-27_21-35-57_checkpoint.pth.tar
