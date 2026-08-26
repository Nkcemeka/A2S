import torch
from models.notagen import Patchilizer
from models.weights_params import get_notagen_params
from notagen_peft import NotagenPeft
from pathlib import Path
from features import *
import time
import tempfile
import music21 as m21
from pyMV2H.utils.music import Music
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.converter.midi_converter import MidiConverter as Converter
import hydra
from omegaconf import DictConfig
from utils.best_checkpoint import get_best_checkpoint

class NotagenEval:
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

    @torch.inference_mode()
    def inference_patch(self, feat, feat_mask, prompt_lines=[]):
        bos_patch = [self.patchilizer.bos_token_id] * (self.PATCH_SIZE - 1) + [\
            self.patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = self.patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))

        prompt_patches = [[ord(c) for c in patch] + [\
            self.patchilizer.special_token_id] * (self.PATCH_SIZE - len(patch)) for patch
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
            return abc_text
        else:
            print('failed')

    def extract_fn(self, feat_name: str, extractor, sample_path: str):
        sample = torch.load(sample_path)     
        # Peak-normalize to [-1, 1] before augmentation
        peak = sample.abs().max()
        
        if peak > 0:
            sample = sample / peak
        if feat_name == "omar":
            feat = extractor.extract(sample)
        elif feat_name == "mfunc":
            # We fix this later....
            raise NotImplementedError()
        elif feat_name == "m2l":
            feat = extractor.extract(sample)
        else:
            raise ValueError(f"{feat_name} not supported!")
        return feat

    def set_extractor(self, feat_type: str, feat_name: str):
        if (feat_type == "midi"):
            extractor = self.midi_features.get(feat_name, None)

        if (feat_type == "audio"):
            extractor = self.audio_features.get(feat_name, None)

        if extractor is None:
            raise RuntimeError(f"No extractor for {feat_name}")

        return extractor

    @torch.inference_mode()
    def inference_sample(self, extractor, sample_path: str, \
            feat_name: str="omar", prompt_lines:list=[]):
        """
            Perform inference by passing the sample.
            Sample can be an audio/MIDI file.

            Args:
            -----
                sample_path (str): Path to sample
                prompt_lines (list): Input prompt lines
                feat_name (str): Name of feature.
        """
        # init the feature extractor
        feat = self.extract_fn(feat_name, extractor, sample_path).squeeze().unsqueeze(0)
        feat_mask = torch.ones(
            feat.shape[:-1],  # [batch, feat-layer, time]
            device=self.device,
            dtype=torch.long,
        )
        return self.inference_patch(feat, feat_mask, prompt_lines)

    def metrics(self, gt_files: list, test_files: list, \
            feat_name: str="omar"):

        extractor = self.set_extractor("audio", feat_name)()
        total_mv2h = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        count = 0

        for gtf, tf in zip(gt_files, test_files):
            pred = self.inference_sample(extractor, tf, feat_name=feat_name)

            gt = Path(gtf).read_text()
            with tempfile.TemporaryDirectory() as tmpdir:
                gt_score = m21.converter.parseData(gt, \
                                format="abc")
                pred_score = m21.converter.parseData(pred, format="abc")

                # remove this later...
                try:
                    pred_score.write("midi", f"{tmpdir}/temp_pred.mid")
                    gt_score.write("midi", f"{tmpdir}/temp_gt.mid")
                except:
                    continue

                gt_midi_path = f"{tmpdir}/temp_gt.mid"
                pred_midi_path = f"{tmpdir}/temp_pred.mid"

                # convert the temp_gt and temp_pred files
                conv_gt = Converter(file=gt_midi_path,\
                        output=gt_midi_path.replace("mid", "txt"))
                conv_pred = Converter(file=pred_midi_path,\
                        output=pred_midi_path.replace("mid", "txt"))

                conv_gt.convert_file()
                conv_pred.convert_file()

                ref_file = Music.from_file(gt_midi_path.replace("mid", "txt"))
                pred_file = Music.from_file(pred_midi_path.replace("mid", "txt"))
                
                try:
                    mv2h_res =  mv2h(ref_file, pred_file)
                    total_mv2h.__multi_pitch__ += mv2h_res.multi_pitch
                    total_mv2h.__voice__ += mv2h_res.voice
                    total_mv2h.__meter__ += mv2h_res.meter
                    total_mv2h.__harmony__ += mv2h_res.harmony
                    total_mv2h.__note_value__ += mv2h_res.note_value
                    count += 1
                except Exception as e:
                    print(f"Skipping {Path(gtf).stem}...")
                    with open(f"{gt_midi_path.replace("mid", "txt")}", "r") as f:
                        lines = f.readlines()
                        print(lines)
                        print("------")

                    with open(f"{pred_midi_path.replace("mid", "txt")}", "r") as f:
                        lines = f.readlines()
                        print(lines)
                        print("------")
                    print(e)
                    return

                print(mv2h_res)

        total_mv2h.__multi_pitch__ /= count
        total_mv2h.__voice__ /= count
        total_mv2h.__meter__ /= count
        total_mv2h.__harmony__ /= count
        total_mv2h.__note_value__ /= count

        mv2h_dict = {
            "Multi-pitch": total_mv2h.__multi_pitch__,
            "Voice": total_mv2h.__voice__,
            "Meter": total_mv2h.__meter__,
            "Harmony": total_mv2h.__harmony__,
            "Note Value": total_mv2h.__note_value__,
            "MV2H": (total_mv2h.__multi_pitch__ + total_mv2h.__voice__ + \
                total_mv2h.__meter__ + total_mv2h.__harmony__ + \
                total_mv2h.__note_value__)/5
        }

        print(mv2h_dict)
        return mv2h_dict

@hydra.main(config_path="configs/conf", \
    config_name="config.yaml", version_base="1.4")
def main(cfg: DictConfig):
    run_dir = Path(__file__).parent / f"runs/{cfg.name}"
    best_ckpt = get_best_checkpoint(run_dir)
    cfg.resume_ckpt = best_ckpt
    eval_obj = NotagenEval(cfg)

    # Get the split directory for the run. This is where the test_split.txt file is located.
    split_dir = Path(__file__).parent / f"runs/{cfg.name}"

    # check if there are paritions in the folder so we can calculate metrics for each partition
    # partitions are folders with the names f1 to f5. If there are no partitions, then 
    # split_dir will be taken to be the partition.
    partitions = [f for f in split_dir.iterdir() if f.is_dir() and f.name.startswith("f")]
    if len(partitions) == 0:
        partitions = [split_dir]

    for partition in partitions:
        test_split_path = partition / "test_split.txt"
        gt_files = []
        test_files = []
        with open(test_split_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                samp_path, abc_path = line.strip().split('\t')
                gt_files.append(abc_path)
                test_files.append(samp_path)

        mv2h_metrics = eval_obj.metrics(gt_files=gt_files, \
            test_files=test_files, feat_name=cfg.feature.feat_name)

        # save the metrics to a file
        print(mv2h_metrics)
        exit(1)
        with open(partition / "mv2h_metrics.txt", "w") as f:
            for key, value in mv2h_metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
