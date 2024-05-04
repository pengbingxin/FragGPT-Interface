import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum

from models import register_model
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertConfig, BertModel
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from einops import rearrange, repeat
from utils.utils import accuracy, accuracy2
from datasets.smiles_data import classification_names, regression_names, tasks, drop_names

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

@register_model(['smiles_fragv2'])
class FragSmilesGPTv2(nn.Module):
    def __init__(self, cfg, task=None, Tokenizer=None):
        super().__init__()
        if task is not None:
            self.tokenizer = task.tokenizer
        elif Tokenizer is not None:
            self.tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        # unimodal decoder
        unimodal_config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            n_layer=cfg.MODEL.UNIMODAL.n_layer,
            n_head=cfg.MODEL.UNIMODAL.n_head,
            n_embd=cfg.MODEL.UNIMODAL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN + 1,
            n_ctx=cfg.DATA.MAX_SMILES_LEN + 1,
        )
        self.unimodal_decoder = GPT2Model(unimodal_config)

        # multimodal decoder
        multimodal_config = GPT2Config(
            n_layer=cfg.MODEL.UNIMODAL.n_layer,
            n_head=cfg.MODEL.UNIMODAL.n_head,
            n_embd=cfg.MODEL.UNIMODAL.n_embd,
        )
        multimodal_config.add_cross_attention = cfg.MODEL.MULTIMODAL.add_cross_attention
        self.multimodal_decoder = MultiModalDecoder(config=multimodal_config)
        # self.multimodal_decoder = GPT2Model(multimodal_config)

        self.to_logits = nn.Sequential(
            LayerNorm(cfg.MODEL.MULTIMODAL.n_embd),
            nn.Linear(cfg.MODEL.MULTIMODAL.n_embd, len(self.tokenizer), bias=False)
        )

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def embed_smiles(self, smiles_ids):
        inputs_embeds = self.unimodal_decoder.wte(smiles_ids)
        attention_mask = smiles_ids.ne(self.frag_padding_idx).type(torch.int64)

        unimodal_output = self.unimodal_decoder(
            inputs_embeds =inputs_embeds,
            attention_mask=attention_mask,
        )

        # cls_embed = unimodal_output['last_hidden_state'][:, -1]
        smiles_embed = unimodal_output['last_hidden_state']

        # get text cls token
        # cls_proj_embed = self.smiles_proj(cls_embed)
        return smiles_embed, attention_mask

    def calc_caption_loss(self, lm_logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        caption_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return caption_loss

    def forward_unimodal(self, input_ids):
        self.frag_padding_idx = self.tokenizer.convert_tokens_to_ids("<pad>")
        smiles_embed, attention_mask = self.embed_smiles(smiles_ids=input_ids)
        return smiles_embed, attention_mask

    def forward_multimodal(self,
                           smiles_embed,
                           attention_mask,
                           encoder_hidden_states=None,
                           encoder_attention_mask=None):

        output = self.multimodal_decoder(
            hidden_states=smiles_embed,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = self.to_logits(output['last_hidden_state'][:, :-1])
        cls = output['last_hidden_state'][:, -1]
        return logits, cls


    def training_loss(self, logits, labels):
        acc = accuracy2(logits[:, :-1], labels[:, 1:])
        caption_loss = self.calc_caption_loss(logits, labels)
        return caption_loss, acc

    def forward(self, input_ids, labels):
        smiles_embed, attention_mask = self.forward_unimodal(input_ids)
        logits, cls = self.forward_multimodal(smiles_embed, attention_mask)
        loss, acc = self.training_loss(logits, labels)
        return {'loss': loss, 'hit@1': acc}

class MultiModalDecoder(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        output_shape = hidden_states.size()
        batch_size = hidden_states.shape[0]

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)

            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            encoder_hidden_states = encoder_hidden_states.contiguous()

        # blocks
        hidden_states = self.drop(hidden_states)
        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states.contiguous(),
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states
        )

if __name__ == '__main__':
    multimodal_config = GPT2Config(
        n_layer=3,
        n_head=8,
        n_embd=768,
    )
    multimodal_config.add_cross_attention = True
    model = MultiModalDecoder(multimodal_config)
    hidden_states = torch.rand(4, 64, 768)
    encoder_hidden_states = torch.rand(4, 10, 768)
    y = model(hidden_states,
        attention_mask=None,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=None,)
    print(y)