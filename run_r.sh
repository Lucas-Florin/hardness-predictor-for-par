python baseline_trainer.py \
    -d rap \
    --gpu-devices 2 \
    --max-epoch 10 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 0 0 \
    --loss-func deepmar \
    --use-macc
