python realistic_predictor_trainer.py \
    -d pa100k \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --load-weights=2019-10-15_11-52-18_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib \
    --rejector none \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.1
    --use-confidence
