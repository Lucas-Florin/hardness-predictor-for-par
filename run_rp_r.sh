taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 4 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --train-batch-size 64 \
    --ap-baseline=../rp_rap/2020-03-28_20-27-02_checkpoint.pth.tar \
    --hp-calib linear \
    --hp-calib-thr mean \
    --main-net-train-epochs 0 \
    --load-weights=../baseline_rap/2020-03-27_21-35-57_checkpoint.pth.tar