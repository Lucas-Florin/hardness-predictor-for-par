python realistic_predictor_analyze.py \
    -d pa100k \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --hard-att Back \
    --num-save-hard 20 \
    --num-save-easy 20 \
    --show-pos-samples
    --plot-acc-hp \
    --plot-pos-hp \


