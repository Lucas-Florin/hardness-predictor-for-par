python realistic_predictor_analyze.py \
    -d rap \
    --gpu-devices 1 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --reject-hard-attributes-quantile 0.5
    --hard-att attachment-Backpack \
    --num-save-hard 20 \
    --num-save-easy 20 \
    --show-neg-samples
    --plot-acc-hp \
    --plot-pos-hp \


