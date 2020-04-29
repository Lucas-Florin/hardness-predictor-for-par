taskset -c 30-34 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 6 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --train-batch-size 64 \
    --ap-baseline=../rp_rap/2020-03-28_20-27-02_checkpoint.pth.tar \
    --hp-calib none \
    --hp-calib-thr mean \
    --use-bbs \
    --hp-visibility-weight 1.0 \
    --main-net-train-epochs 0 \
    --load-weights=../baseline_rap/2020-03-27_21-35-57_checkpoint.pth.tar