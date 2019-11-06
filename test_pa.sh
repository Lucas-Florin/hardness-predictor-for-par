python baseline_trainer.py \
    -d pa100k \
    --gpu-devices 3 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --load-weights=2019-06-07_22-18-59_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib