taskset -c 20-24 \
python realistic_predictor_trainer.py \
    -d rap \
    --hp-model resnet50_nh_strong \
    --gpu-devices 4 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/baseline_rap/ \
    --load-weights=2020-07-03_11-10-40_checkpoint.pth.tar \
    --evaluate \
    --ap-baseline=2020-03-31_12-56-57_checkpoint.pth.tar
