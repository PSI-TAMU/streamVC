#!/bin/bash

python preprocess/preprocess_data.py data -c experiments/base/config.json

python preprocess/pitch_norm.py data