"""
    Extracts features for audio files
    using OMAR-RQ!
"""
from omar_rq import get_model
import torchaudio
import numpy as np
import torch
from pathlib import Path
import gin

@gin.configurable
class OMAR:
    def __init__(self, model_id="mtg-upf/omar-rq-multifeature-25hz-fsq"):
        self._model_id = model_id
        self.sr_map = {
            "mtg-upf/omar-rq-multifeature-25hz-fsq": 24000,
            "mtg-upf/omar-rq-multifeature-25hz": 24000,
            "mtg-upf/omar-rq-multifeature": 24000,
            "mtg-upf/omar-rq-multicodebook": 16000,
            "mtg-upf/omar-rq-base": 16000,
            "mtg-upf/omar-rq-base-freesound-small": 16000,
            "mtg-upf/omar-rq-base-freesound-large": 16000,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(model_id=self._model_id, device=self.device)

    @torch.inference_mode()
    def extract(self, audio: str | torch.Tensor, layers: list =[]):
        """
            Extracts omar-rq features.

            Args:
                audio (str | torch.Tensor): Path to audio file or loaded
                             audio tensor
                layers (list): List of layers to return. If empty, it
                               returns all by default.

            Returns:
                features (torch.Tensor): Omar-RQ features
        """
        if isinstance(audio, str):
            x, sr = torchaudio.load(audio)
            x = x.to(self.device)
            if x.size(0) > 1:
                x = x.mean(dim=0, keepdim=True)

            if sr != self.sr_map[self._model_id]:
                print(f"{sr} not equal to OMAR's SR: {\
                    self.sr_map[self._model_id]}. Resampling...")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sr_map[self._model_id]
                ).to(x.device)

                x = resampler(x)
        else:
            x = audio
            x = x.to(self.device)

        peak = x.abs().max()
        if peak > 0:
            x = x/peak

        if layers:
            # check the maximum layer 
            max_layer = max(layers)
            if max_layer > 24:
                raise ValueError("Total number of layers cannot exceed 24!")
            embeddings = self.model.extract_embeddings(x, layers=layers)
        else:
            layers = np.arange(0, 24).tolist()
            embeddings = self.model.extract_embeddings(x, layers=layers)
        return embeddings

    def save_features(self, audio_path: str, layers: list, output_path: str, suffix: str|None=None):
        embeddings = self.extract(audio_path, layers).squeeze()
        stem = Path(audio_path).stem

        if suffix is None:
            torch.save(embeddings.detach().cpu(), str(Path(output_path)/f"{stem}.pt"))
        else:
            torch.save(embeddings.detach().cpu(), str(Path(output_path)/f"{stem}_{suffix}.pt"))
