import pickle
import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from tqdm import tqdm

import src.synthesizer as synthesizer
from evaluation.dnsmos.dnsmos_local import ComputeScore as DNSScorer


config = "experiments/base/config.json"
ckpt = "experiments/base/g_01120000"
device = "cuda"

synthesizer.load_models(config, ckpt, device)


# DNS MOS Model Setup
p808_model_path = os.path.join('/mnt/nvme-data1/waris/CortexCraft/streamingVC/SynthesisEvaluation/dnsmos/DNSMOS', 'model_v8.onnx')
primary_model_path = os.path.join('/mnt/nvme-data1/waris/CortexCraft/streamingVC/SynthesisEvaluation/dnsmos/DNSMOS', 'sig_bak_ovr.onnx')
dns_scorer = DNSScorer(primary_model_path, p808_model_path)


with open('/mnt/nvme-data1/waris/CortexCraft/SpeakerAttribute/data/composite_pc_gender_age.pkl', 'rb') as f:
    data = pickle.load(f)

# print(data.keys())
composite_pc_gender = data['correlation_weight_gender']
composite_pc_age = data['correlation_weight_age']

spkr_names = ["RMS", "SLT", "BDL", "CLB"]
lambda_embeds = np.arange(-1000, 1001, 10)
root_folder = "/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


for spkr_name in tqdm(spkr_names):
    wav_paths = list(Path(f'{root_folder}/{spkr_name}/wav').glob('*.wav'))
    wav_paths = random.sample(wav_paths, 100)
    lamda_for_wavs = []
    for wav_path in tqdm(wav_paths):
        lamda_for_wav = []
        #Spkr Embed
        embed = synthesizer.get_speaker_embed(wav_path)
        for lambda_embed in lambda_embeds:
            new_embed = embed + np.float32(lambda_embed) * np.float32(composite_pc_gender)

            with TemporaryDirectory() as tmp_dir:
                out_path = Path(tmp_dir) / f"{spkr_name}_{wav_path.stem}_{lambda_embed}.wav"
                _, _ = synthesizer.process_wav_stream(wav_path, new_embed, out_path, chunk_size=160)

                synth_embed = synthesizer.get_speaker_embed(out_path)
                cosine_sim = cosine_similarity(embed, synth_embed)

                mos_ = dns_scorer(out_path, 16000, False)
                lamda_for_wav.append([cosine_sim, mos_['P808_MOS']])

        lamda_for_wavs.append(lamda_for_wav)

    lamda_for_wavs = np.array(lamda_for_wavs)
    rows = []
    for i, lambda_embed in enumerate(lambda_embeds):
        row = {}
        row['lambda'] = lambda_embed
        row['cosine_sim'] = f"{np.mean(lamda_for_wavs[:, i, 0])} ± {np.std(lamda_for_wavs[:, i, 0])}"
        row['P808_MOS'] = f"{np.mean(lamda_for_wavs[:, i, 1])} ± {np.std(lamda_for_wavs[:, i, 1])}"
        rows.append(row)

    df = pd.DataFrame(rows)
    # csv_path = Path(f"{spkr_name}")
    df.to_csv((Path('attribute_editing/results') / f"{spkr_name}_lambda_embeds.csv"))


            
