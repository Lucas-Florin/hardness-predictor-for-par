python realistic_predictor_analyze.py \
    -d market1501 \
    --gpu-devices 0 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_market/ \
    --num-save-hard 20 \
    --num-save-easy 20 \
    --hard-att handbag \
    --plot-acc-hp \
    --show-pos-samples


