from pathlib import Path
import re

def get_best_checkpoint(directory):
    checkpoints = Path(directory).glob("*.ckpt")

    best = None
    best_loss = float("inf")

    for path in checkpoints:
        match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", path.name)

        if match is None:
            continue

        loss = float(match.group(1))
        
        if loss < best_loss:
            best_loss = loss
            best = path

    return best
