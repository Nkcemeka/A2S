import torch
import random
import bisect
import re
import numpy as np
from transformers import GPT2LMHeadModel, PreTrainedModel
from .gpt2_model import GPT2Model as YieldGPT2Model
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from .yield_tags import Tags
from .weights_params import get_notagen_params
from .adapter import Adapter

class Patchilizer:
    def __init__(self, model_type:str="large"):
        self.params = get_notagen_params(model_type)
        self.stream = self.params["PATCH_STREAM"]
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1] 
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, generate_last=False):
        patch_size = self.params["PATCH_SIZE"]
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes
        

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)

        return metadata_patches
    
    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
       
        return tunebody_patches

    def encode_train(self, abc_text, add_special_patches=True, cut=True):
        patch_length = self.params["PATCH_LENGTH"]
        patch_size = self.params["PATCH_SIZE"]
        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break

        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]    # [r:n/n]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]
            #tunebody_patches = [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                raise RuntimeError("Too long....")
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))  
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index) 
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index] 
                    stream_tunebody_lines = tunebody_lines[line_index : ]
                    
                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut:
            if len(patches) > patch_length:
                raise RuntimeError("Don't cut!")
            patches = patches[ : patch_length]
        else:
            pass

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code, add_special_patches=True):
        patch_length = self.params["PATCH_LENGTH"]
        patch_size = self.params["PATCH_SIZE"]
        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
    
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
    
        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < patch_size and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches

    def encode_validate(self, abc_text, add_special_patches=True, cut=True):
            patch_length = self.params["PATCH_LENGTH"]
            patch_size = self.params["PATCH_SIZE"]
            lines = abc_text.split('\n')
            lines = list(filter(None, lines))
            lines = [line + '\n' for line in lines]
    
            tunebody_index = -1
            for i, line in enumerate(lines):
                if line.startswith('[V:'):
                    tunebody_index = i
                    break
    
            metadata_lines = lines[ : tunebody_index]
            tunebody_lines = lines[tunebody_index : ]
    
            if self.stream:
                tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                    enumerate(tunebody_lines)]    # [r:n/n]
    
            metadata_patches = self.patchilize_metadata(metadata_lines)
            tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')
    
            if add_special_patches:
                bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
                eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)
    
                metadata_patches = [bos_patch] + metadata_patches
                tunebody_patches = tunebody_patches + [eos_patch]
    
            patches = metadata_patches + tunebody_patches
    
            if cut:
                if len(patches) > patch_length:
                    raise RuntimeError("Don't cut!")
                patches = patches[ : patch_length]
            else:
                pass
    
            # encode to ids
            id_patches = []
            for patch in patches:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
                id_patches.append(id_patch)
    
            return id_patches
    

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)

class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config, patch_size):
        super().__init__(config)
        self.patch_size = patch_size
        self.patch_embedding = torch.nn.Linear(patch_size * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = YieldGPT2Model(config)
        self.n_layer = config.num_hidden_layers
        self.keep_prob = 0.25

        # learned embedding for mask embedding
        self.mask_embedding = torch.nn.Parameter(
            torch.randn(config.n_embd)
        )

    def forward(self,
                patches: torch.Tensor,
                masks=None, cond_dict: dict|None=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.

        Args:
        -----
            patches (torch.Tensor): the patches to be encoded
            masks (torch.Tensor | None): the masks for the patches
            cond_dict (dict | None): Condition dictionary

        Returns:
        --------
            result: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, self.patch_size * (128))

        # define the patches generator
        patches = self.patch_embedding(patches.to(self.device))

        # # Add some masked embedding
        # if self.training:
        #     B, L, D = patches.shape

        #     # keep True = preserve history
        #     keep = (
        #         torch.rand(B, L, device=patches.device)
        #         < self.keep_prob
        #     )

        #     # Never mask BOS
        #     keep[:, 0] = True

        #     # Never mask the patch currently being predicted
        #     keep[:, -1] = True

        #     mask_embedding = self.mask_embedding.view(1, 1, D)

        #     patches = torch.where(
        #         keep.unsqueeze(-1),
        #         patches,
        #         mask_embedding
        #     )

        if masks==None:
            gen = self.base(inputs_embeds=patches)
        else:
            gen = self.base(inputs_embeds=patches,
                             attention_mask=masks)

        if cond_dict is None:
            while True:
                try:
                    next(gen)
                except StopIteration as e:
                    return e.value

        # if condition dictionary is not None, we do cross-attn with 
        # adapters
        adapters = cond_dict["adapters"]
        layer_sym_aud_map = cond_dict["layer_sim_aud_map"]
        feat = cond_dict["feat"] # (B, L, seq_len, embed_dim)
        
        if feat.dim() == 3:
            feat = feat.unsqueeze(1)

        feat_mask = cond_dict["feat_mask"] # (B, L, seq_len)

        # create attn mask to deal with padding
        # attn_mask = (feat_mask.unsqueeze(1).unsqueeze(1).to(feat.dtype))
        # attn_mask = ((1.0 - attn_mask)* torch.finfo(feat.dtype).min)

        adapter_out = [None, None]
        while True:
            try:
                data = next(gen)
                sym_layer = data[-1]

                # Check if this layer is something we are
                # interested in
                if sym_layer not in layer_sym_aud_map:
                    continue

                # Get the adapter layer that sym_layer maps to
                # adapter_layer = layer_sym_aud_map[sym_layer]
                adapter_idx, audio_layer = layer_sym_aud_map[sym_layer]
                adapter = adapters[adapter_idx]

                # access the tag and check if it is an hidden state
                # or prenorm output
                tag = data[0]
                if tag == Tags.HIDDEN_STATES:
                    # if it is an hidden state, pass hidden state 
                    # through the corresponding adapter
                    adapter: Adapter = adapters[adapter_idx]

                    attn_mask = (feat_mask[:, audio_layer].unsqueeze(1).unsqueeze(1).to(feat.dtype))
                    attn_mask = ((1.0 - attn_mask)* torch.finfo(feat.dtype).min)

                    # store the result and the sym_layer
                    kv = feat[:, audio_layer] # serves as key and value
                    nq = data[1].shape[1] # seq_len for query
                    nk = kv.shape[1] # seq_len for key
                    indices_query = torch.arange(nq, device=data[1].device)
                    indices_key = torch.arange(nk,device=kv.device)
                    res_adapter = adapter(query=data[1], key=kv, value=kv, attn_mask=attn_mask, \
                        indices_query=indices_query, indices_key=indices_key)
                    adapter_out = [res_adapter, sym_layer]
                elif tag == Tags.PRENORM_OUTPUT:
                    # if it is a prenorm, we should have the 
                    # adapter output for this sym layer
                    if adapter_out[-1] != sym_layer:
                        raise AssertionError(f"The adapter information stored is for layer {\
                            adapter_out[-1]} rather than layer {sym_layer}.")
                    data[1] += adapter_out[0]
            except StopIteration as e:
                return e.value

class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the chars within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config, patch_sampling_batch_size: int=0):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.patch_sampling_batch_size = patch_sampling_batch_size

        self.base = GPT2LMHeadModel(config)

        # mask embedding
        self.mask_embedding = torch.nn.Parameter(
            torch.randn(config.n_embd)
        )

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.bos_token_id, target_patches), dim=1)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if self.patch_sampling_batch_size!=0 and self.patch_sampling_batch_size<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:self.patch_sampling_batch_size])

            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # Add some masked embedding
        # if self.training:
        #     L, T, D = inputs_embeds.shape # seq_len, num_tokens, dim

        #     # keep True = preserve history
        #     keep = (
        #         torch.rand(L, T, device=inputs_embeds.device)
        #         < 0.95
        #     )

        #     # Never mask BOS
        #     keep[:, 0] = True

        #     # Never mask the patch currently being predicted
        #     keep[:, -1] = True

        #     mask_embedding = self.mask_embedding.view(1, 1, D)

        #     inputs_embeds = torch.where(
        #         keep.unsqueeze(-1),
        #         inputs_embeds,
        #         mask_embedding
        #     )

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)
        output = self.base(inputs_embeds=inputs_embeds, 
                         attention_mask=target_masks,
                         labels=labels)
        return output

    def generate(self,
                 encoded_patch: torch.Tensor,
                 tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)
        return probs

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

class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen is a language model with a hierarchical structure.
    It includes a patch-level decoder and a char-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The char-level decoder is used to generate the chars within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, decoder_config, params: dict):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        # Get parameters for model
        self.params = params
        self.patch_level_decoder = PatchLevelDecoder(encoder_config, self.params["PATCH_SIZE"])
        self.char_level_decoder = CharLevelDecoder(decoder_config, 0)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor, cond_dict: dict|None=None):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        masks = masks.clone()
        patches = patches.reshape(len(patches), -1, self.params["PATCH_SIZE"])
        encoded_patches = self.patch_level_decoder(patches, masks,\
                         cond_dict=cond_dict)["last_hidden_state"]

        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]
        return self.char_level_decoder(encoded_patches, patches)

    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        patch_size = self.params["PATCH_SIZE"]
        if patches.shape[-1] % patch_size != 0:
            tokens = patches[:,:,-(patches.shape[-1]%patch_size):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%patch_size)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, patch_size)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []            

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = safe_normalize_probs(prob)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = safe_normalize_probs(prob)
            token = temperature_sampling(prob, temperature=temperature)
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= self.params["PATCH_SIZE"]:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch
