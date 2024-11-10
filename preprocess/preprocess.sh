python preprocess/pitch_calculation.py /mnt/nvme-data1/waris/ProcessedData/IndicTTS/native/metadata/data.tsv \
                                    -c /mnt/nvme-data1/waris/CortexCraft/streamingVC/experiments/french_base/config.json

python preprocess/pitch_norm.py /mnt/nvme-data1/waris/datasets/IndicTTS

python preprocess/energy_calculation.py /mnt/nvme-data1/waris/ProcessedData/IndicTTS/native/metadata/data.tsv \
                                    -c /mnt/nvme-data1/waris/CortexCraft/streamingVC/experiments/french_base/config.json