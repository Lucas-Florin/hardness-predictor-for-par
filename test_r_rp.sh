python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 0 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --load-weights=2019-06-28_15-43-51_checkpoint.pth.tar \
    --evaluate \
    --use-macc \
    --f1-calib \
    --num-save-hard 20 \
    --num-save-easy 20 \
    --hard-att hs-LongHair