#!/bin/bash

common_params="--dataset CWRU --unbiased --sample-rate 12000"

rm -r data
python3 main.py --cfg cfgs/resnet18-spec.yaml --run PARAMETER-TESTING-SPECTROGRAM ${common_params}
rm -r data
python3 main.py --cfg cfgs/resnet18-scalo.yaml --run PARAMETER-TESTING-SCALOGRAM ${common_params}
rm -r data
python3 main.py --cfg cfgs/resnet18-pwv.yaml --run PARAMETER-TESTING-PWV ${common_params}
rm -r data
python3 main.py --cfg cfgs/resnet18-all.yaml --run PARAMETER-TESTING-ALL ${common_params}
