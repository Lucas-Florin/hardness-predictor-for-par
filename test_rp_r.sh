taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 4 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_rap/ \
    --load-weights=2020-04-28_02-06-40_checkpoint.pth.tar \
    --evaluate \
    --ap-baseline=2020-03-28_20-27-02_checkpoint.pth.tar
