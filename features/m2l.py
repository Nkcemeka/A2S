from music2latent import EncoderDecoder
import librosa
from pathlib import Path
import torch

class Music2Latent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encdec = EncoderDecoder(device=device)

    def extract(self, audio_path: str):
        wv, sr = librosa.load(audio_path, sr=44100, mono=True)
        features = self.encdec.encode(wv, extract_features=True)
        features = features.transpose(1, 2)
        return features.squeeze()

    def save_features(self, audio_path: str, output_dir: str, suffix: str|None=None):
        embeddings = self.extract(audio_path)
        stem = Path(audio_path).stem

        if suffix is None:
            torch.save(embeddings.detach().cpu(), str(Path(output_dir)/f"{stem}.pt"))
        else:
            torch.save(embeddings.detach().cpu(), str(Path(output_dir)/f"{stem}_{suffix}.pt"))
