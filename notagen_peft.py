import pytorch_lightning as pl
from transformers import GPT2Config
from models.notagen import NotaGenLMHeadModel
import torch
from models.weights_params import get_notagen_params
from peft import get_peft_model
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from features import *
import hydra

def safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return probs

class NotagenPeft(pl.LightningModule):
    def __init__(self, feat_name: str="omar", model_type:str='large', use_lora_char:bool=False,\
             use_lora_patch:bool=False, lora_cfg: dict=None, map_cfg: dict=None, \
            adapter_cfg: dict=None, pair_loss_flag: bool=True, pair_loss_margin: float=0.05):
        super().__init__()
        params = get_notagen_params(model_type)
        PATCH_NUM_LAYERS = params["PATCH_NUM_LAYERS"]
        PATCH_LENGTH = params["PATCH_LENGTH"]
        HIDDEN_SIZE = params["HIDDEN_SIZE"]
        CHAR_NUM_LAYERS = params["CHAR_NUM_LAYERS"]
        PATCH_SIZE = params["PATCH_SIZE"]
        INFERENCE_WEIGHTS_PATH = params["INFERENCE_WEIGHTS_PATH"]

        # initialize Notagen and load the pretrained weights
        patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                            max_length=PATCH_LENGTH, 
                            max_position_embeddings=PATCH_LENGTH,
                            n_embd=HIDDEN_SIZE,
                            num_attention_heads=HIDDEN_SIZE//64,
                            vocab_size=1)
        char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                                    max_length=PATCH_SIZE+1, 
                                    max_position_embeddings=PATCH_SIZE+1,
                                    hidden_size=HIDDEN_SIZE,
                                    num_attention_heads=HIDDEN_SIZE//64,
                                    vocab_size=128)
        self.base_model = NotaGenLMHeadModel(encoder_config=patch_config,\
             decoder_config=char_config, params=params)
        checkpoint = torch.load(INFERENCE_WEIGHTS_PATH)
        self.base_model.load_state_dict(checkpoint['model'], strict=False)

        # Freeze the base model's weights and
        # Freeze the entire model
        for p in self.base_model.parameters():
            p.requires_grad = False

        if map_cfg is None or lora_cfg is None or adapter_cfg is None:
            raise RuntimeError("Please provide a valid Lora cfg and Map cfg through hydra!")

        # Apply LORA if possible
        if use_lora_patch:
            lora_config = hydra.utils.instantiate(lora_cfg)
            self.base_model.patch_level_decoder.base = get_peft_model(\
                self.base_model.patch_level_decoder.base, lora_config)

        if use_lora_char:
            lora_config = hydra.utils.instantiate(lora_cfg)
            self.base_model.char_level_decoder.base = get_peft_model(\
                            self.base_model.char_level_decoder.base, lora_config)

        # make the mask embedding for char decoder trainable
        #self.base_model.char_level_decoder.mask_embedding.requires_grad = True

        map_config = hydra.utils.instantiate(map_cfg)
        self.map_dict = map_config.get_dict()
        num_adapters = len(self.map_dict)
        self.adapters = nn.ModuleList([hydra.utils.instantiate(adapter_cfg) for _ in range(num_adapters)])
        self.audio_features = {"omar": OMAR, "m2l": Music2Latent}

        if feat_name not in self.audio_features:
            raise RuntimeError(f"Supported features include: `omar` and `m2l`!")

        self.extractor = self.audio_features[feat_name]()
        self.pair_loss_flag = pair_loss_flag
        self.pair_loss_margin = pair_loss_margin

    def forward(self, feat: torch.Tensor, feat_mask: torch.Tensor, \
            patches: torch.Tensor, patches_masks: torch.Tensor):

        cond_dict = {
            "layer_sim_aud_map": self.map_dict,
            "feat": feat,
            "feat_mask": feat_mask,
            "adapters": self.adapters
        }

        out = self.base_model(patches, patches_masks, cond_dict=cond_dict)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.base_model.params["LR"],
        )
        return optimizer

    def _pair_loss(self, feat, feat_mask, patches, patches_masks):
        batch_size = patches.shape[0]
        seed = torch.randint(0, 2**32, (), device=feat.device).item()
        torch.manual_seed(seed)
        out = self.forward(feat, feat_mask, patches, patches_masks)

        if batch_size < 2:
            # This could happen for validation
            # wrong loss is same as normal loss, but pair loss is 0 here...
            # or we return margin???
            return out, out, torch.tensor(
                0.0, device=feat.device
            )

        if self.pair_loss_flag is False:
            return out, out, torch.tensor(
                0.0, device=feat.device
            )

        shift = torch.randint(1, batch_size, (), device=feat.device)
        wrong_indices = (
            torch.arange(batch_size, device=feat.device) + shift
        ) % batch_size

        torch.manual_seed(seed)
        wrong_out = self.forward(
            feat[wrong_indices],
            feat_mask[wrong_indices],
            patches,
            patches_masks.clone(),
        )

        loss = out.loss
        wrong_loss = wrong_out.loss
        pair_loss = F.relu(self.pair_loss_margin + loss - wrong_loss)
        return out, wrong_out, pair_loss

    def training_step(self, train_batch, batch_idx):
        samples_batch, samples_len_batch, patches, patches_masks = train_batch
        feat_ext = []

        max_seq_len = 0
        for i, (sample, sample_len) in enumerate(zip(samples_batch, samples_len_batch)):
            sample = sample[:int(sample_len.item())].unsqueeze(0)
            extract = self.extractor.extract(sample).squeeze().unsqueeze(0)
            feat_ext.append(extract)
            max_seq_len = max(max_seq_len, extract.shape[-2])

        # pad as necessary
        feat = []
        feat_mask = []
        for each in feat_ext:
            mask = torch.ones((1, each.shape[-3], each.shape[-2])) # (B, L, seq_len)
            mask = F.pad(mask, (0, max_seq_len-each.shape[-2]),\
                            mode="constant", value=0).to(self.device)
            each = F.pad(each, (0, 0, 0, max_seq_len-each.shape[-2]),\
                mode="constant", value=0)
            feat.append(each)
            feat_mask.append(mask)

        feat = torch.cat(feat)
        feat_mask = torch.cat(feat_mask)

        out, wrong_out, pair_loss = self._pair_loss(feat, feat_mask, patches, patches_masks)
        loss = out.loss
        wrong_loss = wrong_out.loss

        total_loss = loss + pair_loss
        self.log_dict(
            {
                'train_loss': loss.item(),
                'train/audio_gap': (wrong_loss - loss).item(),
                'train_pair_loss': pair_loss.item(),
                'total_train_loss': total_loss.item(),
            }, logger=True, on_step=False, on_epoch=True
        )

        for i, adapter in enumerate(self.adapters):
            self.log_dict({
                f'adapter_gate/adapter_{i}': adapter.cross_attn.gates
            })
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        samples_batch, samples_len_batch, patches, patches_masks = val_batch
        feat_ext = []

        max_seq_len = 0
        for i, (sample, sample_len) in enumerate(zip(samples_batch, samples_len_batch)):
            sample = sample[:int(sample_len.item())].unsqueeze(0)
            extract = self.extractor.extract(sample).squeeze().unsqueeze(0)
            feat_ext.append(extract)
            max_seq_len = max(max_seq_len, extract.shape[-2])

        # pad as necessary
        feat = []
        feat_mask = []
        for each in feat_ext:
            mask = torch.ones((1, each.shape[-3], each.shape[-2])) # (B, L, seq_len)
            mask = F.pad(mask, (0, max_seq_len-each.shape[-2]),\
                            mode="constant", value=0).to(self.device)
            each = F.pad(each, (0, 0, 0, max_seq_len-each.shape[-2]),\
                mode="constant", value=0)
            feat.append(each)
            feat_mask.append(mask)

        feat = torch.cat(feat)
        feat_mask = torch.cat(feat_mask)
        out, wrong_out, pair_loss = self._pair_loss(feat, feat_mask, patches, patches_masks)
        loss = out.loss
        wrong_loss = wrong_out.loss

        self.log_dict(
            {
                'val_loss': loss.item(),
                'val/audio_gap': (wrong_loss - loss).item(),
                'val_pair_loss': pair_loss.item(),
            }, logger=True, on_step=False, on_epoch=True
        )
        return loss

    @torch.inference_mode()
    def generate(self,
                 feat, feat_mask,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0, greedy=True):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        # top_k = 1
        # top_p = 0.5
        # temperature = 1.0
        # greedy = True

        PATCH_SIZE = self.base_model.params["PATCH_SIZE"]
        cond_dict = {
                    "layer_sim_aud_map": self.map_dict,
                    "feat": feat,
                    "feat_mask": feat_mask,
                    "adapters": self.adapters
        }

        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.base_model.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.base_model.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE) # [bs, seq, patch_size]
        encoded_patches = self.base_model.patch_level_decoder(patches, cond_dict=cond_dict)["last_hidden_state"]    # [bs, seq, hidden_size]
        generated_patch = []

        while True:
            prob = self.base_model.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            if greedy:
                token = prob.argmax(axis=-1)
            else:
                prob = safe_normalize_probs(prob)
                prob = top_k_sampling(prob, top_k=top_k, return_probs=True) # [128]
                prob = safe_normalize_probs(prob)
                prob = top_p_sampling(prob, top_p=top_p, return_probs=True) # [128]
                prob = safe_normalize_probs(prob)
                token = temperature_sampling(prob, temperature=temperature) # int
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            elif len(tokens)>=2 and tokens[-2].item() == 1 and tokens[-1].item() == 2:
                # This means we have reached the EOS patch
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)

        return generated_patch
