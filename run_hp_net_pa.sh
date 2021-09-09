python realistic_predictor_trainer.py \
    -d pa100k \
    --max-epoch 30 \
    --save-experiment=./experiments/rp_pa100k/ \
    --optim-group-pretrained \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --fix-seed 

