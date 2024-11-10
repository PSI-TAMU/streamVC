import os
import torch
import torchaudio
import librosa
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from argparse import RawTextHelpFormatter
from multiprocessing.pool import ThreadPool
from pathlib import Path

from src.utils import attr_dict
from src.dataset import mel_spectogram_sb


def get_spk_wavs(dataset_path, output_path):
    wav_files = []
    os.makedirs(f"{output_path}", exist_ok=True)
    for spks in os.listdir(dataset_path):
        if os.path.isdir(f"{dataset_path}/{spks}"):
            os.makedirs(f"{output_path}/{spks}", exist_ok=True)
            for file in os.listdir(f"{dataset_path}/{spks}"):
                if file.endswith(".wav"):
                    wav_files.append(f"{dataset_path}/{spks}/{file}")
        elif spks.endswith(".wav"):
            wav_files.append(f"{dataset_path}/{spks}")
    return wav_files

def get_spk_wavs_tsv(tsv_path):
    wav_files = open(tsv_path, "r").readlines()
    wav_files = [file.strip().split("\t")[0] for file in wav_files]
    return wav_files

def calculate_energy(wav, wavs_folder, energy_folder, h, exist_ok=False):
    audio, fs = torchaudio.load(wav)
    # Computes energy
    if energy_folder is not None:
        energy_file = wav.replace(".wav", ".pit").replace(
            wavs_folder, energy_folder
        )
    else:
        (Path(wav).parent.parent / 'energy').mkdir(parents=True, exist_ok=True)
        energy_file = Path(wav).parent.parent / 'energy' / f"{Path(wav).stem}.eng"

    if exist_ok and os.path.isfile(energy_file):
        return
    else:
        audio, _ = librosa.load(wav, sr = 16000)
        audio = torch.from_numpy(audio)
        _, energy = mel_spectogram_sb(audio, h.sampling_rate, h.code_hop_size, h.win_size, 
                                      h.n_fft, h.num_mels, h.fmin, h.fmax)
        np.save(energy_file, energy)

def extract_energy(wav_files, dataset_path, output_path, args, concurrency):
    bound_process_wav = partial(calculate_energy, wavs_folder=dataset_path, energy_folder=output_path, h=args)

    with ThreadPool(concurrency) as pool:
        list(tqdm(pool.imap(bound_process_wav, wav_files), total=len(wav_files)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each wav file in a dataset.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset waves.")
    parser.add_argument(
        "--output_path", type=str, help="path for output speaker/speaker_wavs.npy."
    )
    parser.add_argument("-c", "--config", help="config file", dest="config", default='experiments/exp1/config.json')
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", 
                        dest="thread_count", type=int, default=4)
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    process_num = args.thread_count

    if os.path.isdir(args.dataset_path):
        output_path = args.output_path
        wav_files = get_spk_wavs(dataset_path, output_path)
    else:
        wav_files = get_spk_wavs_tsv(dataset_path)
        output_path = None
    
    h = attr_dict(args.config)
    extract_energy(wav_files, dataset_path, output_path, h, process_num)

"""
python preprocess/pitch_calculation.py /home/jupyter/data_clean/waves-16k /home/jupyter/data_clean/pitch -c /home/jupyter/CoAC/experiments/exp2/config.json
"""