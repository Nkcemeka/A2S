"""
    Extracts features from
    midi-alignment paper.
"""
from models.midi_function_alignment.cp_transformer_fine_tune \
    import RoFormerSymbolicTransformerInjected
from models.midi_function_alignment.yield_tags import Tags
import torch
from pathlib import Path
from tqdm import tqdm
import gin

@gin.configurable
class MFunc:
    def __init__(self, ckpt_path: str, midi_pt: str, base_dir: str, max_position_embeddings: int=768):
        self.base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(ckpt_path, \
            max_position_embeddings=max_position_embeddings, strict=False)
        self.n_layers = self.base_model.num_layers
        self.device = self.base_model.device
        self.midi_pt = midi_pt
        self.base_dir = base_dir

    def extract_features(self, output_dir:str|None=None):
        assert str(Path(self.midi_pt).parent) == str(Path(self.base_dir)),\
            f"MIDI pt parent directory should be same as base_dir"
        self.extract(self.midi_pt, self.base_dir, output_dir)

    def extract(self, midi_pt:str, BASE_DIR:str, output_dir:str|None):
        if output_dir is None:
            FEATURE_DIR = Path(BASE_DIR) / "features"
        else:
            FEATURE_DIR = output_dir
            
        file_path = midi_pt
        data = torch.load(file_path, weights_only=True)
        pitch_shift_range = torch.load(file_path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
        pitch_shift_range = torch.zeros_like(pitch_shift_range) # no pitch shift for us
        length = torch.load(file_path[:-3] + '.length.pt', weights_only=True)
        start = torch.cumsum(length, dim=0) - length

        with open(file_path[:-3] + '.txt') as f:
            piece_info = f.readlines()

        midi_filenames = []
        for idx in range(len(piece_info)):
            line = piece_info[idx].strip()
            _, midi_path = line.split("\t")
            midi_path = str(Path(BASE_DIR) / ("midi/" + midi_path))
            assert Path(midi_path).exists(), f"{midi_path} does no exist!"
            midi_filenames.append(midi_path)

        assert len(midi_filenames) == len(start), "Number of MIDI files not equal to processed info."

        # Create features_dir if it does not exist
        Path(FEATURE_DIR).mkdir(exist_ok=True, parents=True)
        for i, (s, dur) in tqdm(enumerate(zip(start, length)),\
                total=len(start), desc="Extracting features..."):
            start_idx = s
            end_idx = s+dur
            chunk = data[start_idx:end_idx]
            chunk_ps = torch.tensor([0], dtype=torch.long).unsqueeze(0).unsqueeze(0)

            # save the chunk and chunk_ps maybe???
            # or we can reconstruct pitch_shift_range based on the shape? yes..
            # pitch shift for chunk is pitch_shift_range[i] which is [0, 0]
            
            # so we pass this chunk through the model
            # preprocess chunk
            chunk = self.base_model.preprocess(chunk.unsqueeze(0), pitch_shift=chunk_ps)
            gen = self.base_model(chunk.to(self.device))
            feat = next(gen); assert feat[0] == Tags.SIMUNOTE_EMBEDDING
            feat = next(gen); assert feat[0] == Tags.PE_POSITIONS
            feat_layers = []
            for layer in range(self.n_layers):
                feat = next(gen); assert feat[0] == Tags.HIDDEN_STATES
                feat_layers.append(feat[1])
                feat = next(gen); assert feat[0] == Tags.PRENORM_OUTPUT

            feat_layers = torch.cat(feat_layers, dim=0) # (L, seq_len, hidden_size)
            stem = Path(midi_filenames[i]).stem
            feat_path = Path(FEATURE_DIR) / (stem + ".pt")
            torch.save(feat_layers.detach().cpu(), str(feat_path))

    def extract_sample(self):
        pass
