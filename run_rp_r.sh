python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split val \
    --rejector-thresholds-split val \
    --random-translation \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback
    --use-deepmar-for-hp