import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PeftDataset, collate_batch_samples
from pathlib import Path
from sklearn.utils import shuffle
import torch

class PeftNottA2SDataModule(pl.LightningDataModule):
    def __init__(self, samples_dir: str, abc_dir: str, partition_dir: str, \
            nott_samples_dir: str, nott_abc_dir: str, hours: float, \
            percent_split: float=0.9, partition_num: int=1, sax_type: str="alto", \
            train_id: list = ["real"], val_id: str="real", \
            test_id: str="real", theta: list= [1.0], aug_flag: bool=False, aug_cfg=None, \
            batch_size: int=16, num_workers: int=4, store_split_dir: str|None=None):
        super().__init__()

        for id in train_id:
            if id not in ["real", "mididdsp", "fluidsynth"]:
                raise ValueError(f"Accepted id values are: `real`, `mididdsp`, and `fluidsynth`.")

        for id in [test_id, val_id]:
            if id not in ["real", "mididdsp", "fluidsynth"]:
                raise ValueError(f"Accepted id values are: `real`, `mididdsp`, and `fluidsynth`.")

        if sax_type not in ["alto", "tenor"]:
            raise ValueError(f"Accepted sax_type values are: `alto` and `tenor`.")

        self.train_id = train_id
        self.val_id = val_id
        self.test_id = test_id

        self.split_dir_hash = {
            "train": [str(Path(samples_dir) / tid / sax_type) for tid in self.train_id],
            "val": str(Path(samples_dir) / self.val_id / sax_type),
            "test": str(Path(samples_dir) / self.test_id / sax_type)
        }
        self.abc_dir = abc_dir
        self.sax_type = sax_type
        self.theta = theta
        self.partition_dir = partition_dir
        self.nott_samples_dir = nott_samples_dir
        self.nott_abc_dir = nott_abc_dir
        self.hours = hours
        self.percent_split = percent_split
        self.partition_num = partition_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.store_split_dir = store_split_dir
        self.aug_flag = aug_flag
        self.aug_cfg = aug_cfg

    def get_split_files(self, partition_dir: str, partition_num: int, split: str):
        split_stems = self.get_stems(partition_dir, partition_num, split)
        split_files_samples = []
        split_files_abc = []
        split_dir = self.split_dir_hash[f"{split}"]

        if isinstance(split_dir, str):
            for stem in split_stems:
                feat_path = Path(split_dir) / f"{stem}.pt"
                split_files_samples.append(feat_path)
                split_files_abc.append(str(Path(self.abc_dir) / self.sax_type / f"{stem}.abc"))
            return split_files_samples, split_files_abc

        # this is a training split, so we treat it differently...
        num_train_subsets = len(split_dir)
        for idx in range(num_train_subsets):
            proportion = int(len(split_stems) * self.theta[idx])
            split_stem_cpy = shuffle(split_stems, random_state=42)[:proportion]
            folder = split_dir[idx]
            for stem in split_stem_cpy:
                feat_path = Path(folder) / f"{stem}.pt"
                abc_path = str(Path(self.abc_dir) / self.sax_type / f"{stem}.abc")
                if not Path(abc_path).exists():
                    # should we handle this, though????
                    continue
                split_files_samples.append(feat_path)
                split_files_abc.append(abc_path)
        return split_files_samples, split_files_abc

    def get_stems(self, partition_dir: str, partition_num: int, split: str):
        split_stem_path = Path(partition_dir) / f"f{partition_num}/{split}.txt"
        with open(split_stem_path, "r") as f:
            lines = f.readlines()

        split_stems = []
        for line in lines:
            stem = Path(line.strip().split('\t')[0]).stem
            split_stems.append(stem)

        return split_stems

    def setup(self, stage):
        self.set_split()

    def get_nottingham(self):
        nott_samples = sorted(Path(self.nott_samples_dir).glob("*.pt"))
        nott_abc = sorted(Path(self.nott_abc_dir).glob("*.abc"))

        # For the nottingham dataset, we want to extract a 
        # certain number of hours
        total_duration = 0.0
        final_nott_samples = []
        final_nott_abc = []

        for samp_path, abc_path in zip(nott_samples, nott_abc):
            # We get the number of hours from the samples file
            # whose sample rate is 24000Hz
            samp_tensor = torch.load(samp_path)
            duration = samp_tensor.shape[1] / 24000.0
            if total_duration + duration <= (self.hours * 3600.0):
                total_duration += duration
                final_nott_samples.append(str(samp_path))
                final_nott_abc.append(str(abc_path))
            else:
                break
        return final_nott_samples, final_nott_abc

    def set_split(self):
        # Define dataset class here
        train_files_samples, train_files_abc = self.get_split_files(
            partition_dir=self.partition_dir, partition_num=self.partition_num,
            split="train"
        )

        val_files_samples, val_files_abc = self.get_split_files(
                    partition_dir=self.partition_dir, partition_num=self.partition_num,
                    split="val"
        )

        test_files_samples, test_files_abc = self.get_split_files(
                    partition_dir=self.partition_dir, partition_num=self.partition_num,
                    split="test"
        )

        # Get nottingham dataset files
        nott_files_samples, nott_files_abc = self.get_nottingham()

        # Combine the nottingham dataset with the training, validation, and test splits
        # with self.percent_split taken for training and the rest for just validation
        nott_split_len = int(self.percent_split * len(nott_files_samples))
        train_files_samples += nott_files_samples[:nott_split_len]
        train_files_abc += nott_files_abc[:nott_split_len]
        val_files_samples += nott_files_samples[nott_split_len:]
        val_files_abc += nott_files_abc[nott_split_len:]
        
        self.train_dataset = PeftDataset(samples_list=train_files_samples,\
                abc_list=train_files_abc, train=True, augment=self.aug_flag, cfg=self.aug_cfg)
        self.val_dataset = PeftDataset(samples_list=val_files_samples,\
                abc_list=val_files_abc, train=False, augment=False, cfg=self.aug_cfg)

        print(f"Number of training files: ", len(self.train_dataset))
        print(f"Number of validation files: ", len(self.val_dataset))

        # save the split files if store_split_dir is provided
        # so in store_split_dir, create a partition folder and store
        # the train, val, and test splits in it
        if self.store_split_dir:
            Path(self.store_split_dir).mkdir(parents=True, exist_ok=True)
            partition_folder = Path(self.store_split_dir) / f"f{self.partition_num}"
            partition_folder.mkdir(parents=True, exist_ok=True)

            with open(partition_folder / "train_split.txt", 'w') as f:
                for samp_path, abc_path in zip(train_files_samples, train_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")
            with open(partition_folder / "val_split.txt", 'w') as f:
                for samp_path, abc_path in zip(val_files_samples, val_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")
            with open(partition_folder / "test_split.txt", 'w') as f:
                for samp_path, abc_path in zip(test_files_samples, test_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")
                    
    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, \
            batch_size=self.batch_size, num_workers=self.num_workers, \
            collate_fn=collate_batch_samples, shuffle=True)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, \
            batch_size=self.batch_size, num_workers=self.num_workers, \
            collate_fn=collate_batch_samples, shuffle=False)
        return dl
