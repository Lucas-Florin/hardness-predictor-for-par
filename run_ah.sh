python train_baseline.py \
    -d pa100k \
    --gpu-devices 2 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --stepsize 60 120 \
    --use-macc \
    --color-jitter \
    --eval-freq 60

python train_baseline.py \
    -d pa100k \
    --gpu-devices 2 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --stepsize 60 120 \
    --use-macc \
    --color-aug \
    --eval-freq 60

python train_baseline.py \
    -d pa100k \
    --gpu-devices 2 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --stepsize 60 120 \
    --use-macc \
    --random-erase \
    --eval-freq 60