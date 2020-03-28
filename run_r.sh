python baseline_trainer.py \
    -d rap \
    --gpu-devices 7 \
    --max-epoch 180 \
    --stepsize 60 120 \
    --eval-split test \
    --eval-freq 20 \
    --save-experiment=./storage/baseline_rap/ \
    --loss-func deepmar
