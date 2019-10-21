python baseline_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --random-translation \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc