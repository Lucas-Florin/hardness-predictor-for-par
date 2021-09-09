python realistic_predictor_trainer.py \
    -d pa100k \
    --max-epoch 30 \
    --save-experiment=./experiments/pa100k/ \
    --optim-group-pretrained \
    --use-deepmar-for-hp \
    --fix-seed 