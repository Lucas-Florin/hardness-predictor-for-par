python realistic_predictor_trainer.py \
    -d rap \
    --max-epoch 30 \
    --save-experiment=./experiments/rp_rap/ \
    --optim-group-pretrained \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --hp-calib linear \
    --hp-calib-thr f1 \
    --f1-baseline=../baseline_rap/***_checkpoint.pth.tar  \
    --fix-seed 

