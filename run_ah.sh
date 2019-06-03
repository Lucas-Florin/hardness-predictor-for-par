

python train_baseline.py \
    -d market1501 \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --stepsize 20 40 \
    --print-freq 100 \
    --group-atts \
    --eval-freq 30 \
    --color-jitter

python train_baseline.py \
    -d market1501 \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --stepsize 20 40 \
    --print-freq 100 \
    --group-atts \
    --eval-freq 30 \
    --color-aug


python train_baseline.py \
    -d market1501 \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --stepsize 20 40 \
    --print-freq 100 \
    --group-atts \
    --eval-freq 30 \
    --random-erase

python train_baseline.py \
    --root datasets \
    -d rap \
    --gpu-devices 0 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 60 120 \
    --loss-func scel \
    --use-macc \
    --eval-freq 30 \
    --color-aug


python train_baseline.py \
    --root datasets \
    -d rap \
    --gpu-devices 0 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 60 120 \
    --loss-func scel \
    --use-macc \
    --eval-freq 30 \
    --random-erase