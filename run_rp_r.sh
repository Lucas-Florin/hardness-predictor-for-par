python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 60 120 \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp
    --load-weights=2019-06-03_17-42-43_checkpoint.pth.tar \
    --train-hp-only