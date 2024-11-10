import librosa
import glob
from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

if __name__ == "__main__":
    max_wav_value = 32768.0
    files = glob.glob("/mnt/nvme-data1/waris/datasets/LibriTTS_O/train-*/**/**/*.wav")
    outdir = Path("/mnt/nvme-data1/waris/streamVC_data/data")
    for file in tqdm(files):
        wav, sr = librosa.load(file, sr=16000)
        wav = wav / max(abs(wav)) * max_wav_value

        fpath = Path(file)
        fid = fpath.stem
        spkr = fpath.parent.parent.name
        (outdir / spkr / 'wav').mkdir(parents=True, exist_ok=True)

        wavfile.write(
            (outdir / spkr / 'wav' / f'{fid}.wav'),
            sr,
            wav.astype(np.int16),
        )

