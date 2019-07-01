python realistic_predictor_trainer.py \
    -d market1501 \
    --gpu-devices 0 \
    --max-epoch 60 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --stepsize 20 40 \
    --loss-func deepmar \
    --group-atts \
    --f1-calib


python realistic_predictor_trainer-.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 180 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --stepsize 60 120 \
    --loss-func scel \
    --use-macc \
    --f1-calib \
    --eval-freq 30