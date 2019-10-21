python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector macc \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.20
    --use-confidence

python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector median \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.20
    --use-confidence

python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector threshold \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.20 \
    --use-confidence

python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector threshold \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.35 \
    --use-confidence

python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector quantile \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.20 \
    --use-confidence

python realistic_predictor_trainer.py \
    -d rap \
    --gpu-devices 1 \
    --max-epoch 20 \
    --stepsize 0 0 \
    --main-net-finetuning-epochs 20 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --loss-func deepmar \
    --use-macc \
    --no-hp-feedback \
    --use-deepmar-for-hp \
    --load-weights=2019-09-19_15-31-32_checkpoint.pth.tar \
    --rejector macc \
    --max-rejection-quantile 0.25 \
    --rejection-threshold 0.20 \
    --use-confidence