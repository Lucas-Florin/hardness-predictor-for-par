# A Hardness Predictor for Pedestrian Attribute Recognition

## Setup
Create conda environment. 
```
conda config --set channel_priority strict
conda env create --file ./environment.yml
conda activate par-hp
```

## Datasets
Put datasets in the `./datasets/` directory: 
The RAP 2.0 dataset in `./datasets/RAP/` and the PA100k dataset in `./datasets/pa100k/` . 

## Replicate Results
**Note:** In Linux, scripts have to be declared as executables before executing: `chmod +x script.sh` 
### PAR Classificator
Train baseline PAR classificators.
```
./run_baseline_rap.sh
./run_baseline_pa.sh
```

Train PAR classificators with hardness score feedback.
```
./run_baseline_with_hp_feedback_rap.sh
./run_baseline_with_hp_feedback_pa.sh
```

### Hardness Predictor
**Note:** To train the HP-Net without DeepMAR weighting remove `--use-deepmar-for-hp` from the shell scripts. 

Train harndess predictor in parallel. 
```
./run_hp_net_rap.sh
./run_hp_net_pa.sh
```

Train hardness predictor sequentially. 
The path to the trained baseline PAR classificator has to be specified in the `--load-weights` option in the script. 
```
./run_hp_net_sequential_rap.sh
./run_hp_net_sequential_pa.sh
```

Train hardness predictor with calibration. 
The path to the trained baseline PAR classificator has to be specified in the `--f1-baseline` option in the script. 
```
./run_hp_net_calibrated_rap.sh
./run_hp_net_calibrated_pa.sh
```

Train hardness predictor with visibility as ground truth (bounding boxes). 
```
./run_hp_net_bbs_rap.sh
```

After training a hardness predictor you can see easy and hard to classify example images:
```
./show_hard_examples_rap.sh
./show_hard_examples_pa.sh
```

## Citation
L. Florin, A. Specker, A. Schumann and J. Beyerer, "Hardness Prediction for More Reliable Attribute-based Person Re-identification," 2021 IEEE 4th International Conference on Multimedia Information Processing and Retrieval (MIPR), 2021, pp. 418-424, doi: 10.1109/MIPR51284.2021.00077.

Paper: https://ieeexplore.ieee.org/abstract/document/9565485