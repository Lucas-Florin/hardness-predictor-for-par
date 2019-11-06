python realistic_predictor_trainer.py \
    -d pa100k \
    --gpu-devices 2 \
    --eval-split val \
    --rejector-thresholds-split train \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --load-weights=2019-10-28_10-28-13_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib \
    --rejector none \
    --max-rejection-quantile 0.1 \
    --rejection-threshold 0.1
    --use-confidence
