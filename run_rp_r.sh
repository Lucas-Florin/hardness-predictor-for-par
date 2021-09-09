taskset -c 10-14 \
python realistic_predictor_trainer.py \
    -d rap \
    --max-epoch 30 \
    --stepsize 15 20 25 \
    --save-experiment=./storage/rp_rap/ \
    --optim-group-pretrained \
    --no-hp-feedback \
    --ap-baseline=../rp_rap/2020-07-09_00-24-24_checkpoint.pth.tar \
    --use-deepmar-for-hp \
    --fix-seed \
    --train-batch-size 32
    --load-weights=../baseline_rap/2020-07-08_20-03-59_checkpoint.pth.tar \
    --main-net-train-epochs 0

