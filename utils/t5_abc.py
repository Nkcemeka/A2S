""" 
    Utility script to convert T5 cataloging dataset from JSONL format to ABC format.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import re

def filter_abc(abc: str):
    lines = abc.splitlines()
    clean_lines = []

    for line in lines:

        if line.startswith("V:"):
            line = re.sub(r'\s+nm="[^"]*"', '', line)
            line = re.sub(r'\s+snm="[^"]*"', '', line)

        # Remove chord annotations such as "G"
        line = re.sub(r'"[^"]*"', '', line)

        clean_lines.append(line)

    return '\n'.join(clean_lines)

def clean_abc(abc: str):
    """ 
        Parse the abc file and remove
        redundant catalog information
        and tempo markings.

        Args:
        ----
            abc (str): ABC file
        
        Returns:
        -------
            clean_abc_text (str): Clean ABC file
    """
    lines = abc.split('\n')
    for i, line in enumerate(lines):
        if line[0] == 'L':
            start = i
            break

    clean_lines = lines[start:]
    return filter_abc('\n'.join(clean_lines))

def main(t5_dir: str, t5_size: int, seed: int, t5_abc: str):
    base_dir = str(Path(t5_dir))
    splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl'}
    df = pd.read_json(f"{base_dir}/" + splits["train"], lines=True)
    df = df[df["task"] == "cataloging"] # filter the dataset based on the catalog subset
    abc_samples = df.sample(t5_size, random_state=seed)

    # iterate through the samples and convert each to MusicXML
    Path(t5_abc).mkdir(parents=True, exist_ok=True)
    for i, sample in tqdm(abc_samples.iterrows(), total=t5_size):
        abc = sample["input"]
        abc = clean_abc(abc)
        out_file = Path(t5_abc) / f"sample_{i}.abc"
        with open(str(out_file), "w") as f:
            f.write(abc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_dir', default="/home/nkcemeka/Documents/Datasets/MelodyT5")
    parser.add_argument('--t5_size', default=5000)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--t5_abc', default="/home/nkcemeka/Documents/Datasets/MelodyT5/t5_abc")

    args = parser.parse_args()
    main(t5_dir=args.t5_dir, t5_size=args.t5_size, seed=args.seed, \
         t5_abc=args.t5_abc)

