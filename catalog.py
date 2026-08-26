import hydra
from omegaconf import DictConfig
from pathlib import Path
from utils.general import windowed_segments, windowed_segments_cumulative
import random
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch

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

def extract_excerpts(xml_dir: str, exceprts_xml_dir: str, \
    excerpts_aud_dir: str, excerpts_midi_dir: str, \
    excerpts_abc_dir: str, excerpts_abc_il_dir: str,\
    synthesis: str, prog: str, sr: float, flag_midi: bool, \
    flag_abc: bool, flag_abc_il: bool, flag_audio: bool, \
    muse_path: str, sf2_path: str, strip_flag: bool, \
    cumulative: bool=False, audio_length: float|None=None, \
    num_measures: list[int]=[5], part_idx: int|None=None):
    """ 
        Extracts excerpts for a given dataset of musicxml files. 
        The excerpts are extracted using a windowed approach. 
        The function can also handle cumulative excerpts, where 
        the excerpts are extracted in a cumulative manner, with 
        each excerpt containing all previous excerpts.

        Args:
            xml_dir (str): Path to the directory containing the musicxml files.
            exceprts_xml_dir (str): Path to the directory where the extracted musicxml excerpts will be saved.
            excerpts_aud_dir (str): Path to the directory where the extracted audio excerpts will be saved.
            excerpts_midi_dir (str): Path to the directory where the extracted midi excerpts will be saved.
            excerpts_abc_dir (str): Path to the directory where the extracted abc excerpts will be saved.
            excerpts_abc_il_dir (str): Path to the directory where the extracted abc_il excerpts will be saved.
            synthesis (str): The synthesis method to be used for audio generation (e.g., "fluidsynth").
            prog (str): The program number to be used for synthesis.
            sr (float): The sample rate for audio generation.
            flag_midi (bool): Whether to generate midi excerpts.
            flag_abc (bool): Whether to generate abc excerpts.
            flag_abc_il (bool): Whether to generate abc_il excerpts.
            flag_audio (bool): Whether to generate audio excerpts.
            muse_path (str): Path to the MuseScore executable for audio generation.
            sf2_path (str): Path to the soundfont file for audio generation.
            strip_flag (bool): Whether to strip repeats in the excerpts.
            cumulative (bool): Whether to extract cumulative excerpts. Default is False.
            audio_length (float|None): The length of the audio excerpts in seconds. Default is None, which means the length will be determined by the window size.
            num_measures (list[int]): List of possible window sizes (in measures) for the excerpts. Default is [5].

        Returns:
            None. The extracted excerpts are saved to the specified directories.
    """

    xml_files = sorted(Path(xml_dir).glob("*.musicxml"))
    for i, xml in enumerate(xml_files):
        random.seed(42)  # Ensure reproducibility for random choices
        measures_excerpt = random.choice(num_measures)
        if not cumulative:
            windowed_segments(score_file=str(xml), out_dir_xml=exceprts_xml_dir, \
                out_dir_abc_il=excerpts_abc_il_dir, out_dir_abc=excerpts_abc_dir, \
                out_dir_midi=excerpts_midi_dir, out_dir_audio=excerpts_aud_dir, \
                window_size=measures_excerpt, hop_size=measures_excerpt, \
                flag_midi=flag_midi, flag_abc=flag_abc, \
                flag_abc_il=flag_abc_il, flag_audio=flag_audio, MUSESCORE_PATH=muse_path,\
                sf2_path=sf2_path, prog=prog, sample_rate=sr, strip_flag=strip_flag, \
                part_idx=part_idx)
        else:
            if audio_length is None:
                audio_length = 20

            windowed_segments_cumulative(score_file=str(xml), out_dir_xml=exceprts_xml_dir, \
                out_dir_abc_il=excerpts_abc_il_dir, out_dir_abc=excerpts_abc_dir, \
                out_dir_midi=excerpts_midi_dir, out_dir_audio=excerpts_aud_dir, \
                flag_midi=flag_midi, flag_abc=flag_abc, \
                flag_abc_il=flag_abc_il, flag_audio=flag_audio, MUSESCORE_PATH=muse_path,\
                sf2_path=sf2_path, prog=prog, sample_rate=sr, strip_flag=strip_flag, \
                audio_length=audio_length, part_idx=part_idx)

@hydra.main(
    config_path="configs/catalog_conf",
    config_name="config",
    version_base="1.3"
)
def main(cfg: DictConfig):
    # Get the excerpts
    extract_excerpts(
        xml_dir=cfg.dataset.XML_DIR,
        exceprts_xml_dir=cfg.dataset.EXCERPTS_XML_DIR,
        excerpts_midi_dir=cfg.dataset.EXCERPTS_MIDI_DIR,
        excerpts_aud_dir=cfg.dataset.EXCERPTS_AUDIO_DIR,
        excerpts_abc_dir=cfg.dataset.EXCERPTS_ABC_DIR,
        excerpts_abc_il_dir=cfg.dataset.EXCERPTS_ABC_IL_DIR,
        synthesis=cfg.dataset.SYNTHESIS,
        prog=cfg.dataset.PROG,
        sr=cfg.dataset.SAMPLE_RATE,
        flag_midi=cfg.dataset.FLAG_MIDI,
        flag_abc=cfg.dataset.FLAG_ABC,
        flag_abc_il=cfg.dataset.FLAG_ABC_IL,
        flag_audio=cfg.dataset.FLAG_AUDIO,
        muse_path=cfg.dataset.MUSESCORE_PATH,
        sf2_path=cfg.dataset.SOUNDFONT,
        strip_flag=cfg.dataset.STRIP_REPEAT,
        cumulative=cfg.dataset.CUMULATIVE,
        audio_length=cfg.dataset.AUDIO_LENGTH,
        part_idx=cfg.dataset.PART_IDX,
        num_measures=cfg.dataset.NUM_MEASURES
    )

if __name__ == "__main__":
    main()
