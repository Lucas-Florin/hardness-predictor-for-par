python realistic_predictor_trainer.py \
    -d rap \
    --max-epoch 30 \
    --save-experiment=./experiments/rp_rap/ \
    --optim-group-pretrained \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --use-bbs-gt \
    --fix-seed 

