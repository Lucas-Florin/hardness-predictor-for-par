taskset -c 10-14 \
python realistic_predictor_trainer.py \
    -d pa100k \
    --gpu-devices 2 \
    --max-epoch 30 \
    --stepsize 15 20 25 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_pa100k/ \
    --loss-func deepmar \
    --optim-group-pretrained \
    --no-hp-feedback \
    --ap-baseline=../rp_pa100k/_checkpoint.pth.tar \
    --fix-seed \
    --train-batch-size 32

