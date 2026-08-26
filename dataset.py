import torch
from torch.utils.data import Dataset
from models.notagen import Patchilizer
from utils.general import postprocess_abc
from features import *
from audiomentations import (
    Compose,
    SevenBandParametricEQ,
    AddGaussianSNR,
    PitchShift,
    RoomSimulator,
    Gain
)
from omegaconf import DictConfig

def collate_batch_samples(input_batches):
    input_sample, input_sample_length, input_gt, input_gt_mask = zip(*input_batches)

    # Pad GT and GT mask normally (they are 1D or 2D sequences)
    input_gt = torch.nn.utils.rnn.pad_sequence(
        input_gt, batch_first=True, padding_value=0
    )
    input_gt_mask = torch.nn.utils.rnn.pad_sequence(
        input_gt_mask, batch_first=True, padding_value=0
    )

    # ---- Pad input_sample: shape [1, dim] ----
    max_seq_len = max(x.shape[-1] for x in input_sample)

    padded_sample = torch.zeros(len(input_sample), max_seq_len)
    padded_sample_lengths = torch.tensor(input_sample_length)

    for i, s in enumerate(input_sample):
        seq_len = s.shape[-1]
        padded_sample[i, :seq_len] = s
    return padded_sample, padded_sample_lengths, input_gt, input_gt_mask

class Augmentation:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Define augmentations
        self.augmentObj = Compose([
            # EQ #1 — independently applied with p=0.5
            SevenBandParametricEQ(
                min_gain_db=self.cfg.eq1.min_gain_db,
                max_gain_db=self.cfg.eq1.max_gain_db,
                p=self.cfg.eq1.p,
            ),

            # EQ #2 — independently applied with p=0.5
            SevenBandParametricEQ(
                min_gain_db=self.cfg.eq2.min_gain_db,
                max_gain_db=self.cfg.eq2.max_gain_db,
                p=self.cfg.eq2.p,
            ),

            # Gaussian noise — independently applied with p=0.5
            AddGaussianSNR(
                min_snr_db=self.cfg.gaussian_noise.min_snr_db,
                max_snr_db=self.cfg.gaussian_noise.max_snr_db,
                p=self.cfg.gaussian_noise.p,
            ),

            # Pitch shift: +/- 2 cents
            PitchShift(
                min_semitones=self.cfg.pitch_shift.min_semitones,
                max_semitones=self.cfg.pitch_shift.max_semitones,
                p=self.cfg.pitch_shift.p,
            ),

            # Reverb
            RoomSimulator(
                min_target_rt60=self.cfg.reverb.min_target_rt60,
                max_target_rt60=self.cfg.reverb.max_target_rt60,
                calculation_mode="rt60",
                leave_length_unchanged=True,
                p=self.cfg.reverb.p,
            ),

            # Gain
            Gain(
                min_gain_db=self.cfg.gain.min_gain_db,
                max_gain_db=self.cfg.gain.max_gain_db,
                p=self.cfg.gain.p,
            ),
        ])

# Create a PeftDataset class
class PeftDataset(Dataset):
    def __init__(self, cfg: DictConfig, samples_list: list, abc_list: list,\
            train: bool=False, augment: bool=True, sample_rate: float=24000):
        super().__init__()
        self.train = train
        self.abc_files = abc_list
        self.patchilizer = Patchilizer()
        self.samples_list = samples_list
        self.augment = augment

        if self.augment and self.train:
            self.augmentObj = Augmentation(cfg).augmentObj
        else:
            self.augmentObj = None
        self.sample_rate = sample_rate

        assert len(self.abc_files) == len(self.samples_list), \
            "Number of samples not equal to number of abc files!"

    def __len__(self):
        return len(self.abc_files)

    def __getitem__(self, index):
        abc_path = self.abc_files[index]

        # load abc file 
        with open(abc_path, "r") as f:
            abc = f.read()
            abc = postprocess_abc(abc) # just to make sure

        sample_path = self.samples_list[index]
        sample = torch.load(sample_path)

        # Peak-normalize to [-1, 1] before augmentation
        peak = sample.abs().max()
        if peak > 0:
            sample = sample / peak

        # perform augmentation
        if self.augmentObj is not None:
            # Added this after training...but should be for training
            # and not validation actually
            sample = torch.Tensor(self.augmentObj(sample.cpu().detach().numpy(), \
                    sample_rate=self.sample_rate))

        sample_length = sample.shape[-1]
        if self.train:
            gt_bytes = self.patchilizer.encode_train(abc)
        else:
            gt_bytes = self.patchilizer.encode_validate(abc)

        gt_bytes_mask =  [1] * len(gt_bytes)
        gt_bytes = torch.tensor(gt_bytes, dtype=torch.long)
        gt_bytes_mask = torch.tensor(gt_bytes_mask, dtype=torch.long)
        return sample, sample_length, gt_bytes, gt_bytes_mask
