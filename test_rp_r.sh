taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 4 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --load-weights=2020-06-24_16-35-14_checkpoint.pth.tar \
    --evaluate \
    --ap-baseline=2020-03-31_12-56-57_checkpoint.pth.tar
