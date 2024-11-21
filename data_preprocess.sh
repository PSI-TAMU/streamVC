#!/bin/bash

python preprocess/preprocess_data.py /mnt/nvme-data1/waris/streamVC_data/data -c experiments/base/config.json -e

python preprocess/pitch_norm.py /mnt/nvme-data1/waris/streamVC_data/data