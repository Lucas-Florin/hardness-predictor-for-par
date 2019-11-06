python realistic_predictor_analyze.py \
    -d rap \
    --gpu-devices 2 \
    --eval-split val \
    --save-experiment=/net/merkur/storage/deeplearning/users/floluc/baseline_rap/ \
    --select-atts Femal AgeLess16 BodyNormal hs-LongHair hs-Hat ub-Vest attachment-Backpack \
    --plot-hp-hist \
    --plot-x-max 0.05 \
    --save-plot \
    --use-confidence

