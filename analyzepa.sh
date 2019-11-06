python realistic_predictor_analyze.py \
    -d pa100k \
    --gpu-devices 0 \
    --eval-split test \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_pa/ \
    --select-atts Female Hat ShoulderBag Backpack UpperSplice boots \
    --plot-hp-hist \
    --plot-x-max 0.05 \
    --save-plot \
    --use-confidence


