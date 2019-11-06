python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --eval-split val \
    --rejector-thresholds-split train \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-10-23_14-49-33_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --rejector none \
    --max-rejection-quantile 0.10 \
    --rejection-threshold 0.15
    --use-confidence

