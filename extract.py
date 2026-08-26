from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch
import hydra
from omegaconf import DictConfig

def extract_audio(audio_path: str, sample_rate: float, out_dir: str):
    """
        Extracts audio samples from a given audio file and 
        saves them to the specified output directory.

        Args:
            audio_path (str): Path to the audio file.
            sample_rate (float): The sample rate for audio extraction.
            out_dir (str): Path to the directory where the extracted 
                           audio samples will be saved.
    """
    out_file = Path(out_dir) / f"{Path(audio_path).stem}.pt"
    x, sr = torchaudio.load(audio_path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        print(f"{sr} not equal to required sample rate: {\
            sample_rate}. Resampling...")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sample_rate
        )
        x = resampler(x)

    torch.save(x, out_file)

def extract_samples_subsets(audio_dirs: list[str], out_dirs: list[str], \
        sample_rate: float=24000):
    """ 
        Extracts audio samples from a list of audio directories and 
        saves them to corresponding output directories. This is done if,
        for example, you have a dataset with multiple subsets (e.g grouped
        by instrument or genre) and you want to keep that structure.

        Args:
            audio_dirs (list[str]): List of paths to the directories containing the audio files.
            out_dirs (list[str]): List of paths to the directories where the extracted audio samples will be saved. Each output directory corresponds to an input audio directory.
            sample_rate (float): The sample rate for audio extraction. Default is 24000 Hz.
    """
    for audio_dir, out_dir in zip(audio_dirs, out_dirs):
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Get the audio files
        audio_files = sorted(Path(audio_dir).glob("*.wav"))

        if len(audio_files) == 0:
            raise RuntimeError(f"{audio_dir} does not contain audio!")

        # call the extractor
        for file in tqdm(audio_files, total=len(audio_files)):
            extract_audio(str(file), sample_rate, out_dir)

@hydra.main(
    config_path="configs/extract_conf",
    config_name="config",
    version_base="1.3"
)
def main(cfg: DictConfig):
    # extract the audio samples for each subset of the dataset
    extract_samples_subsets(
        audio_dirs=cfg.dataset.AUDIO_DIRS,
        out_dirs=cfg.dataset.OUT_DIRS,
        sample_rate=cfg.dataset.SAMPLE_RATE)


if __name__ == "__main__":
    main()
