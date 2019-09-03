python baseline_trainer.py \
    -d rap \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 20 40 \
    --loss-func deepmar \
    --use-macc
