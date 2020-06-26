taskset -c 20-25 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 4 \
    --max-epoch 10 \
    --stepsize 60 120 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --loss-func deepmar \
    --no-hp-feedback \
    --ap-baseline=../rp_rap/2020-03-28_20-27-02_checkpoint.pth.tar \
    --use-deepmar-for-hp \
    --pretrained-hp \
    --fix-seed