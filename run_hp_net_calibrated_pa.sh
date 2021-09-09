python realistic_predictor_trainer.py \
    -d pa100k \
    --max-epoch 30 \
    --save-experiment=./experiments/rp_pa100k/ \
    --optim-group-pretrained \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --hp-calib linear \
    --hp-calib-thr f1 \
    --f1-baseline=../baseline_pa100k/***_checkpoint.pth.tar  \
    --fix-seed 

