taskset -c 10-14 \
python realistic_predictor_trainer.py \
    -d pa100k \
    --gpu-devices 2 \
    --eval-split test \
    --rejector-thresholds-split val \
    --save-experiment=./storage/rp_pa100k/ \
    --load-weights=2020-07-13_19-16-05_checkpoint.pth.tar \
    --evaluate \
    --ap-baseline=2020-07-14_01-43-46_checkpoint.pth.tar
