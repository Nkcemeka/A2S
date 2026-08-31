from utils.general import rest_unreduce
import torch
from models.notagen import Patchilizer
from models.weights_params import get_notagen_params
from notagen_peft import NotagenPeft
import time
from pathlib import Path
from features import *
import hydra
from omegaconf import DictConfig
from utils.best_checkpoint import get_best_checkpoint

class NotagenInf:
    def __init__(self, cfg: DictConfig):
        model_type = cfg.model_type
        ckpt_path = cfg.resume_ckpt
        device = "cuda" if torch.cuda.is_available else "cpu"
        self.device = device
        params = get_notagen_params(model_type)
        self.PATCH_SIZE = params["PATCH_SIZE"]
        self.TOP_K = params["TOP_K"]
        self.TEMPERATURE = params["TEMPERATURE"]
        self.TOP_P = params["TOP_P"]
        self.PATCH_LENGTH = params["PATCH_LENGTH"]
        self.patchilizer = Patchilizer()
        self.model = NotagenPeft.load_from_checkpoint(checkpoint_path=ckpt_path, \
            feat_name=cfg.feature.feat_name, model_type=model_type, \
            use_lora_patch=cfg.use_lora_patch, use_lora_char=cfg.use_lora_char, lora_cfg=cfg.lora, \
            map_cfg=cfg.mapping, adapter_cfg=cfg.adapter, \
            pair_loss_flag=cfg.pair_loss_flag, pair_loss_margin=cfg.pair_loss_margin)
        
        self.model = self.model.to(device)
        self.model.eval()
        self.audio_features = {"omar": OMAR, "m2l": Music2Latent}
        self.midi_features = {"mfunc": MFunc}

    def inference_patch(self, feat, feat_mask, prompt_lines=[],\
            out_path_org: str|None=None, out_path_inter:str|None=None):
        
        bos_patch = [self.patchilizer.bos_token_id] * (self.PATCH_SIZE - 1) + [self.patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = self.patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [self.patchilizer.special_token_id] * (self.PATCH_SIZE - len(patch)) for patch
                            in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = torch.tensor(prompt_patches, device=self.device).reshape(1, -1)

        failure_flag = False
        end_flag = False
        cut_index = None

        tunebody_flag = False
        while True:
            predicted_patch = self.model.generate(feat, feat_mask, input_patches.unsqueeze(0),
                                                top_k=self.TOP_K,
                                                top_p=self.TOP_P,
                                                temperature=self.TEMPERATURE)
            if not tunebody_flag and self.patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(self.device)
                temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                predicted_patch = self.model.generate(feat, feat_mask, temp_input_patches.unsqueeze(0),
                                                    top_k=self.TOP_K,
                                                    top_p=self.TOP_P,
                                                    temperature=self.TEMPERATURE)
                predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch

            if predicted_patch[0] == self.patchilizer.bos_token_id and predicted_patch[1] == self.patchilizer.eos_token_id:
                end_flag = True
                break

            next_patch = self.patchilizer.decode([predicted_patch])

            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            patch_end_flag = False
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = self.patchilizer.special_token_id
                if predicted_patch[j] == self.patchilizer.eos_token_id:
                    patch_end_flag = True

            predicted_patch = torch.tensor([predicted_patch], device=self.device)  # (1, 16)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            if len(byte_list) > 102400:
                failure_flag = True
                break
            if time.time() - start_time > 20 * 60:  
                failure_flag = True
                break

            if input_patches.shape[1] >= self.PATCH_LENGTH * self.PATCH_SIZE and not end_flag:
                print('Stream generating...')
                abc_code = ''.join(byte_list)
                abc_lines = abc_code.split('\n')

                tunebody_index = None
                for i, line in enumerate(abc_lines):
                    if line.startswith('[r:') or line.startswith('[V:'):
                        tunebody_index = i
                        break
                if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                    break

                metadata_lines = abc_lines[:tunebody_index]
                tunebody_lines = abc_lines[tunebody_index:]

                metadata_lines = [line + '\n' for line in metadata_lines]
                if not abc_code.endswith('\n'):  
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                        tunebody_lines[-1]]
                else:
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                if cut_index is None:
                    cut_index = len(tunebody_lines) // 2

                abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                input_patches = self.patchilizer.encode_generate(abc_code_slice)

                input_patches = [item for sublist in input_patches for item in sublist]
                input_patches = torch.tensor([input_patches], device=self.device)
                input_patches = input_patches.reshape(1, -1)

        if not failure_flag:
            abc_text = ''.join(byte_list)
                    
            # unreduce
            unreduced_output_path = out_path_inter
            
            abc_lines = abc_text.split('\n')
            abc_lines = list(filter(None, abc_lines))
            abc_lines = [line + '\n' for line in abc_lines]

            try:
                abc_lines = rest_unreduce(abc_lines)

                if unreduced_output_path is not None:
                    with open(unreduced_output_path, 'w') as file:
                        file.writelines(abc_lines)
            except:
                pass
            else:
                # original
                original_output_path = out_path_org

                if original_output_path is not None:
                    with open(original_output_path, 'w') as w:
                        w.write(abc_text)
        else:
            print('failed')

    def extract_fn(self, feat_name: str, extractor, audio_path: str):
        if feat_name == "omar":
            feat = extractor.extract(audio_path)
        elif feat_name == "mfunc":
            # We fix this later....
            raise NotImplementedError()
        elif feat_name == "m2l":
            feat = extractor.extract(audio_path)
        else:
            raise ValueError(f"{feat_name} not supported!")
        return feat

    def inference_sample(self, sample_path: str, \
            feat_name: str="omar", prompt_lines:list=[], out_path_org: str|None=None, \
            out_path_inter: str|None=None):
        """
            Perform inference by passing the sample.
            Sample can be an audio/MIDI file.

            Args:
            -----
                sample_path (str): Path to sample
                prompt_lines (list): Input prompt lines
                feat_name (str): Name of feature.
                out_path_org (str): Path to store original ABC
                out_path_inter (str): Path to store interleaved ABC
        """
        suffix = Path(sample_path).suffix
        extractor = None
        if (".mid" in suffix or ".midi" in suffix):
            extractor = self.midi_features.get(feat_name, None)

        if (".wav" in suffix):
            extractor = self.audio_features.get(feat_name, None)

        if extractor is None:
            raise RuntimeError(f"No extractor for {feat_name}")

        # init the feature extractor
        extractor = extractor()
        feat = self.extract_fn(feat_name, extractor, sample_path).squeeze().unsqueeze(0)
        feat_mask = torch.ones(
            feat.shape[:-1],  # [batch, feat-layer, time]
            device=self.device,
            dtype=torch.long,
        )
        self.inference_patch(feat, feat_mask, prompt_lines, out_path_org=out_path_org, \
            out_path_inter=out_path_inter)

@hydra.main(config_path="configs/conf", \
    config_name="config.yaml", version_base="1.4")
def main(cfg: DictConfig):
    run_dir = Path(__file__).parent / f"runs/{cfg.name}/f5"
    best_ckpt = get_best_checkpoint(run_dir)
    cfg.resume_ckpt = best_ckpt
    inf_obj = NotagenInf(cfg)
    if cfg.in_path is None:
        raise RuntimeError("USAGE: python inference.py --config-name <config_name> +in_path=<sample_path>'")

    if cfg.in_path is not None:
        inf_obj.inference_sample(sample_path=cfg.in_path, feat_name=cfg.feature.feat_name)

if __name__ == "__main__":
    main()
