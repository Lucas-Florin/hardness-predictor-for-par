python dataset_analyzer.py \
    -d pa100k \
    --use-cpu \
    --eval-split test \
    --select-att AgeOver60 \
    --num-save-hard 20 \
    --num-save-easy 20 \
    --show-label-examples \