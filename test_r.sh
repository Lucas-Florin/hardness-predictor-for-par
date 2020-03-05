python baseline_trainer.py \
    -d rap \
    --gpu-devices 0 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-05-28_19-43-57_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib