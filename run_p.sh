python baseline_trainer.py \
    -d pa100k \
    --gpu-devices 0 \
    --max-epoch 1 \
    --stepsize 60 120 \
    --eval-split test \
    --eval-freq 20 \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --loss-func scel