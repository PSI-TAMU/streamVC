import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="data tsv path", dest="data", required=True)
    parser.add_argument("-k", "--km", help="data km path", dest="km", required=True)
    parser.add_argument("-e", "--exp_dir", help="experiment directory", dest="edir", required=True)


    args = parser.parse_args()
    print("Data Path: ", args.data)
    print("Km Path: ", args.km)

    os.makedirs(args.edir, exist_ok=True)

    data = open(args.data, 'r').readlines()

    km_data = open(args.km, 'r').readlines()
    assert len(data) == len(km_data)

    all_items = []

    for file, km in tqdm(zip(data, km_data)):
        fpath = file.strip().split('\t')[0]
        fid = Path(fpath).stem
        rootPath = Path(fpath).parent.parent
        path_spk = f"{rootPath}/spkr/{fid}.spk.npy"
        path_wave = f"{rootPath}/wav/ewav/{fid}.wav"
        path_pitch = f"{rootPath}/pitch/{fid}.pit.npy"
        path_energy = f"{rootPath}/energy/{fid}.eng.npy"
        hubert = km.strip()

        has_error = 0
        if not os.path.isfile(path_spk):
            print_error(path_spk)
            has_error = 1
        if not os.path.isfile(path_wave):
            print_error(path_wave)
            has_error = 1
        if not os.path.isfile(path_pitch):
            print_error(path_pitch)
            has_error = 1
        if not os.path.isfile(path_energy):
            print_error(path_energy)
            has_error = 1
        if has_error == 0:
            all_items.append(
                f"{path_wave}|{path_spk}|{path_pitch}|{path_energy}|{hubert}"
            )

    random.seed(1234)
    random.shuffle(all_items)
    trains, valids, _, _ = train_test_split(all_items, all_items, test_size=0.01, random_state=42)
    trains, dev, _, _ = train_test_split(trains, trains, test_size=0.01, random_state=42)

    fw = open(f"{args.edir}/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()

    fw = open(f"{args.edir}/dev.txt", "w", encoding="utf-8")
    for strs in dev:
        print(strs, file=fw)
    fw.close()

    fw = open(f"{args.edir}/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
