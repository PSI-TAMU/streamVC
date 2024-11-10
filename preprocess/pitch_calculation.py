import os
import torch
import torchaudio
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from argparse import RawTextHelpFormatter
from multiprocessing.pool import ThreadPool
from pathlib import Path

from src.utils import attr_dict


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

def calculate_pitch(wav, wavs_folder, pitch_folder, h, exist_ok=False):
    audio, fs = torchaudio.load(wav)
    # Computes pitch
    if pitch_folder is not None:
        pitch_file = wav.replace(".wav", ".pit").replace(
            wavs_folder, pitch_folder
        )
    else:
        (Path(wav).parent.parent / 'pitch').mkdir(parents=True, exist_ok=True)
        pitch_file = Path(wav).parent.parent / 'pitch' / f"{Path(wav).stem}.pit"

    if exist_ok and os.path.isfile(pitch_file):
        return
    else:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform=audio,
            sample_rate=fs,
            frame_time=(h.pitch_hop_length / fs),
            win_length=3,
            freq_low=h.pitch_min_f0,
            freq_high=h.pitch_max_f0,
        ).squeeze(0)

        # Concatenate last element to match duration.
        pitch = torch.cat([pitch, pitch[-1].unsqueeze(0)])

        # # Mean and Variance Normalization
        # mean = 256.1732939688805
        # std = 328.319759158607

        # pitch = (pitch - mean) / std
        np.save(pitch_file, pitch)

def extract_pitch(wav_files, dataset_path, output_path, args, concurrency):
    bound_process_wav = partial(calculate_pitch, wavs_folder=dataset_path, pitch_folder=output_path, h=args)

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
    extract_pitch(wav_files, dataset_path, output_path, h, process_num)

"""
python preprocess/pitch_calculation.py /home/jupyter/data_clean/waves-16k /home/jupyter/data_clean/pitch -c /home/jupyter/CoAC/experiments/exp2/config.json
"""