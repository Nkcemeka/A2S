import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PeftDataset, collate_batch_samples
from pathlib import Path
import random

class PeftDataModule(pl.LightningDataModule):
    def __init__(self, samples_dir: str, abc_dir: str, \
            percent_split: float=0.8, batch_size: int=16, num_workers: int=4, \
            store_split_dir: str|None=None, aug_flag: bool=False, aug_cfg=None):
        super().__init__()
        self.samples_dir = samples_dir
        self.abc_dir = abc_dir
        self.percent_split = percent_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.store_split_dir = store_split_dir
        self.aug_flag = aug_flag
        self.aug_cfg = aug_cfg

    def setup(self, stage):
        self.set_split()

    def set_split(self):
        # Define dataset class here
        abc_files = sorted(Path(self.abc_dir).glob("*.abc"))
        final_samp_files = []
        final_abc_files = []

        for each in abc_files:
            stem = Path(each).stem
            samp_path = Path(self.samples_dir) / f"{stem}.pt"
            if Path(samp_path).exists():
                final_samp_files.append(str(samp_path))
                final_abc_files.append(str(each))

        # Tune-level 80-10-10 split: assign whole tunes to a split so no tune
        # appears in both train and val (windows are named {tune}_{window}).
        tune_of = lambda f: Path(f).stem.rsplit("_", 1)[0]
        tunes = sorted({tune_of(f) for f in final_abc_files})

        random.seed(42)
        random.shuffle(tunes)
        num_tunes = len(tunes)

        print(f"Number of tunes available for training and validation: {num_tunes}")
        train_tunes = tunes[:int(self.percent_split*num_tunes)]
        val_tunes = tunes[int(self.percent_split*num_tunes):]
        len_val_test = len(val_tunes)
        test_tunes = val_tunes[:int(0.5*len_val_test)]
        val_tunes = val_tunes[int(0.5*len_val_test):]

        train_files_abc = [f for f in final_abc_files if tune_of(f) in train_tunes]
        val_files_abc = [f for f in final_abc_files if tune_of(f) in val_tunes]
        test_files_abc = [f for f in final_abc_files if tune_of(f) in test_tunes]
        train_files_samp = [str(Path(self.samples_dir) / f"{Path(f).stem}.pt") for f in train_files_abc]
        val_files_samp = [str(Path(self.samples_dir) / f"{Path(f).stem}.pt") for f in val_files_abc]
        test_files_samp = [str(Path(self.samples_dir) / f"{Path(f).stem}.pt") for f in test_files_abc]

        if self.store_split_dir:
            # create the directory if it doesn't exist
            Path(self.store_split_dir).mkdir(parents=True, exist_ok=True)
            # store as samp_path\tabc_path\n
            with open(Path(self.store_split_dir) / "train_split.txt", 'w') as f:
                for samp_path, abc_path in zip(train_files_samp, train_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")
            with open(Path(self.store_split_dir) / "val_split.txt", 'w') as f:
                for samp_path, abc_path in zip(val_files_samp, val_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")
            with open(Path(self.store_split_dir) / "test_split.txt", 'w') as f:
                for samp_path, abc_path in zip(test_files_samp, test_files_abc):
                    f.write(f"{samp_path}\t{abc_path}\n")

        self.train_dataset = PeftDataset(samples_list=train_files_samp,\
                abc_list=train_files_abc, train=True, augment=self.aug_flag, cfg=self.aug_cfg)
        self.val_dataset = PeftDataset(samples_list=val_files_samp,\
                abc_list=val_files_abc, train=False, augment=False, cfg=self.aug_cfg)

        print(f"Number of training files: ", len(self.train_dataset))
        print(f"Number of validation files: ", len(self.val_dataset))

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
