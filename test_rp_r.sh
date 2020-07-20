taskset -c 30-34 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 6 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --load-weights=2020-07-09_00-24-24_checkpoint.pth.tar \
    --evaluate \
    --ap-baseline=2020-07-09_00-24-24_checkpoint.pth.tar
