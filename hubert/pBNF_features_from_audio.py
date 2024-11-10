import logging
import os
import sys
import joblib
from pathlib import Path

from tqdm import tqdm
import torch
import librosa
import numpy as np
import torch.nn.functional as F

from hubert.dump_pBNF_features import HubertFeatureReader
from hubert.dump_km_labels import ApplyKmeans

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")



# def main(in_dir, text_fpath, out_dir, ckpt_path, layer, km_path, max_chunk):
def main(text_fpath, out_dir, ckpt_path, config, km_path, max_chunk):
    reader = HubertFeatureReader(ckpt_path, config, max_chunk)
    apply_kmeans = ApplyKmeans(km_path)

    text_data = open(text_fpath, 'r').readlines()
    data_split = Path(text_fpath).stem
    lab_path = f"{out_dir}/{data_split}.txt"
    with open(lab_path, "w") as f:
        for line in tqdm(text_data):
            # basename, speaker, _, _ = line.strip("\n").split("|")
            # fpath = os.path.join(in_dir, speaker, 'audio', f"{basename}.aud.npy")
            fpath, spkr, pit, eng, _ = line.strip("\n").split("|")
            feat = reader.get_feats(fpath)
            lab = apply_kmeans(feat.cpu().numpy()).tolist()
            f.write(f"{fpath}|{spkr}|{pit}|{eng}|" + " ".join(map(str, lab)) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("in_dir")
    parser.add_argument("text_fpath")
    parser.add_argument("out_dir")
    parser.add_argument("ckpt_path")
    parser.add_argument("config")
    parser.add_argument("km_path")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))