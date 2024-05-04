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
from datasets.smiles_data import classification_names, regression_names

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

@register_model(['smiles_admet'])
class SmiLES_ADMET(nn.Module):
    def __init__(self, cfg, task, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.caption_loss_weight = cfg.MODEL.CAPTION_LOSS_WEIGHT
        self.contrastive_loss_weight = cfg.MODEL.CONTRASTIVE_LOSS_WEIGHT
        # frag encoder
        frag_config = BertConfig(
            hidden_size=cfg.MODEL.FRAG_MODEL.n_embd,
            num_attention_heads=cfg.MODEL.FRAG_MODEL.n_head,
            num_hidden_layers=cfg.MODEL.FRAG_MODEL.n_layer,
        )
        self.frag_encoder = BertEncoder(frag_config)

        # smiles unimodal decoder
        if tokenizer is None:
            tokenizer = task.tokenizer
        self.tokenizer = tokenizer
        gpt_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.UNIMODAL.n_layer,
            n_head=cfg.MODEL.UNIMODAL.n_head,
            n_embd=cfg.MODEL.UNIMODAL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN+1,
            n_ctx=cfg.DATA.MAX_SMILES_LEN+1,
        )
        self.unimodal_decoder = GPT2Model(gpt_config)
        self.cls_norm = LayerNorm(cfg.MODEL.UNIMODAL.n_embd)

        # admet encoder
        self.admet_cls_token = nn.Embedding(1, self.cfg.MODEL.UNIMODAL.n_embd)
        admet_config = BertConfig(
            hidden_size=cfg.MODEL.ADMET_MODEL.n_embd,
            num_attention_heads=cfg.MODEL.ADMET_MODEL.n_head,
            num_hidden_layers=cfg.MODEL.ADMET_MODEL.n_layer,
        )
        self.admet_encoder = BertModel(admet_config)
        self.admet_encoder_proj = nn.Linear(cfg.MODEL.ADMET_MODEL.n_embd,
                                            cfg.MODEL.ADMET_MODEL.n_embd)

        # query transformer & attn pool  for admet
        self.admet_queries = nn.Parameter(torch.randn(54 + 1, cfg.MODEL.UNIMODAL.n_embd))
        self.admet_attn_pool = CrossAttention(
            dim=cfg.MODEL.UNIMODAL.n_embd,
            context_dim=cfg.MODEL.ADMET_MODEL.n_embd,
            dim_head=cfg.MODEL.ADMET_MODEL.n_embd//cfg.MODEL.ADMET_MODEL.n_head,
            heads=cfg.MODEL.ADMET_MODEL.n_head,
            norm_context=True
        )
        self.admet_attn_pool_norm = LayerNorm(cfg.MODEL.UNIMODAL.n_embd)
        self.type_embed = nn.Embedding(2, cfg.MODEL.ADMET_MODEL.n_embd)

        # multimodal_decoder
        multimodal_config = GPT2Config(
            n_layer=cfg.MODEL.MULTIMODAL.n_layer,
            n_head=cfg.MODEL.MULTIMODAL.n_head,
            n_embd=cfg.MODEL.MULTIMODAL.n_embd,
        )
        multimodal_config.add_cross_attention = True
        self.multimodal_decoder = MultiModalDecoder(config=multimodal_config)

        # logits
        self.to_logits = nn.Sequential(
            LayerNorm(cfg.MODEL.MULTIMODAL.n_embd),
            nn.Linear(cfg.MODEL.MULTIMODAL.n_embd, len(tokenizer), bias=False)
        )

        # contrastive loss
        self.temperature = nn.Parameter(torch.Tensor([1.]))

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(self,
        smiles_ids: torch.LongTensor,
        admet_prop: torch.tensor,
        fragment_ids: torch.LongTensor,
        labels: torch.LongTensor = None,
    ):
        bs = smiles_ids.shape[0]
        # frag encoding
        self.frag_padding_idx = self.tokenizer.convert_tokens_to_ids("<pad>")
        frag_position_ids = create_position_ids_from_input_ids(
            fragment_ids,
            padding_idx=self.frag_padding_idx,
            past_key_values_length=0
        )
        frag_inputs_embeds = self.unimodal_decoder.wte(fragment_ids)
        frag_position_embeddings = self.unimodal_decoder.wpe(frag_position_ids)
        frag_inputs_embeds += frag_position_embeddings
        frag_attention_mask = fragment_ids.ne(self.frag_padding_idx).type(torch.int64)

        frag_attention_mask = frag_attention_mask[:, None, None, :]
        frag_attention_mask = frag_attention_mask.to(dtype=frag_inputs_embeds.dtype)  # fp16 compatibility
        frag_attention_mask = (1.0 - frag_attention_mask) * torch.finfo(frag_inputs_embeds.dtype).min

        frag_hidden = self.frag_encoder(
            hidden_states =frag_inputs_embeds,
            attention_mask=frag_attention_mask,
        ).last_hidden_state

        # unimodal decoding
        smiles_cls_embed, smiles_embed, attention_mask = self.embed_smiles(
            frag_cls  =frag_hidden[:, :1, :],
            smiles_ids=smiles_ids,
            cls_return=True,
        )

        # admet encoding
        admet_cls_token_feature = self.admet_cls_token.weight.unsqueeze(0).repeat(bs, 1, 1)
        admet_prop = admet_prop.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.ADMET_MODEL.n_embd).type(smiles_cls_embed.dtype)
        admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
        type_ids = torch.LongTensor([0 for _ in range(len(classification_names))] +
                                    [1 for _ in range(len(regression_names))]).to(admet_cls_token_feature.device)
        type_embed = self.type_embed(type_ids).type(admet_cls_token_feature.dtype)
        admet_prop_inputs_embd = admet_prop_inputs_embd + type_embed
        admet_prop_inputs_embeds = torch.cat([admet_cls_token_feature, admet_prop_inputs_embd], dim=1)
        admet_hidden = self.admet_encoder(inputs_embeds=admet_prop_inputs_embeds).last_hidden_state

        # admet attention pooling
        admet_queries = repeat(self.admet_queries, 'n d -> b n d', b=admet_hidden.shape[0])
        admet_queries = self.admet_attn_pool(admet_queries, admet_hidden)
        admet_queries = self.admet_attn_pool_norm(admet_queries)
        admet_cls_embed, admet_embed = admet_queries[:, 0], admet_queries[:, 1:]

        # multimodal decoding
        output = self.multimodal_decoder(
            hidden_states         = smiles_embed,
            attention_mask        = attention_mask,
            encoder_hidden_states = admet_embed,
        )
        logits = self.to_logits(output['last_hidden_state'])
        if labels is not None:
            acc = accuracy2(logits[:, :-1], labels[:, 1:])
            caption_loss, contrastive_loss = self.calc_loss(smiles_cls_embed, admet_cls_embed, logits, labels)
            loss = caption_loss + contrastive_loss
            return {
                'loss': loss,
                'caption_loss': caption_loss,
                'contrastive_loss':contrastive_loss,
                'hit@1': acc,
            }
        else:
            return {'logits': logits}

    def embed_smiles(self, frag_cls, smiles_ids, cls_return=True):
        inputs_embeds = self.unimodal_decoder.wte(smiles_ids)
        attention_mask = smiles_ids.ne(self.frag_padding_idx).type(torch.int64)
        inputs_embeds = torch.cat([frag_cls, inputs_embeds[:, 1:, :]], dim=1)

        unimodal_output = self.unimodal_decoder(
            inputs_embeds =inputs_embeds,
            attention_mask=attention_mask,
        )

        if cls_return:
            cls_embed = unimodal_output['last_hidden_state'][:, -1]
            smiles_embed = unimodal_output['last_hidden_state'][:, :-1]

            # get text cls token
            cls_embed = self.cls_norm(cls_embed)
            return cls_embed, smiles_embed, attention_mask[:, :-1]
        else:
            return None, unimodal_output['last_hidden_state'], attention_mask[:, :-1]

    def calc_loss(self, smiles_cls_embed, admet_cls_embed, lm_logits, labels):
        batch, device = smiles_cls_embed.shape[0], smiles_cls_embed.device

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        caption_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        caption_loss = caption_loss * self.caption_loss_weight

        # calculate contrastive loss
        # smiles and admet
        sim = einsum('i d, j d -> i j', smiles_cls_embed, admet_cls_embed)
        sim = sim * self.temperature.exp()

        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (loss_fct(sim, contrastive_labels) + loss_fct(sim.t(), contrastive_labels)) * 0.5

        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss, contrastive_loss

    def generate_once(self,
        smiles_ids: torch.LongTensor,
        admet_prop: torch.tensor,
        fragment_ids: torch.LongTensor,
    ):
        bs = smiles_ids.shape[0]
        # frag encoding
        self.frag_padding_idx = self.tokenizer.convert_tokens_to_ids("<pad>")
        frag_position_ids = create_position_ids_from_input_ids(
            fragment_ids,
            padding_idx=self.frag_padding_idx,
            past_key_values_length=0
        )
        frag_inputs_embeds = self.unimodal_decoder.wte(fragment_ids)
        frag_position_embeddings = self.unimodal_decoder.wpe(frag_position_ids)
        frag_inputs_embeds += frag_position_embeddings
        frag_attention_mask = fragment_ids.ne(self.frag_padding_idx).type(torch.int64)

        frag_attention_mask = frag_attention_mask[:, None, None, :]

        frag_hidden = self.frag_encoder(
            hidden_states=frag_inputs_embeds,
            attention_mask=frag_attention_mask,
        ).last_hidden_state

        # unimodal decoding
        frag_cls = frag_hidden[:, :1, :]
        inputs_embeds = self.unimodal_decoder.wte(smiles_ids)
        attention_mask = smiles_ids.ne(self.frag_padding_idx).type(torch.int64)
        inputs_embeds = torch.cat([frag_cls, inputs_embeds[:, 1:, :]], dim=1)

        unimodal_output = self.unimodal_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        smiles_embed = unimodal_output['last_hidden_state']

        # admet encoding
        admet_cls_token_feature = self.admet_cls_token.weight.unsqueeze(0).repeat(bs, 1, 1)
        admet_prop = admet_prop.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.ADMET_MODEL.n_embd).type(
            smiles_embed.dtype)
        admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
        type_ids = torch.LongTensor([0 for _ in range(len(classification_names))] +
                                    [1 for _ in range(len(regression_names))]).to(admet_cls_token_feature.device)
        type_embed = self.type_embed(type_ids).type(admet_cls_token_feature.dtype)
        admet_prop_inputs_embd = admet_prop_inputs_embd + type_embed
        admet_prop_inputs_embeds = torch.cat([admet_cls_token_feature, admet_prop_inputs_embd], dim=1)
        admet_hidden = self.admet_encoder(inputs_embeds=admet_prop_inputs_embeds).last_hidden_state

        # admet attention pooling
        admet_queries = repeat(self.admet_queries, 'n d -> b n d', b=admet_hidden.shape[0])
        admet_queries = self.admet_attn_pool(admet_queries, admet_hidden)
        admet_queries = self.admet_attn_pool_norm(admet_queries)
        admet_cls_embed, admet_embed = admet_queries[:, 0], admet_queries[:, 1:]

        # multimodal_decoder
        output = self.multimodal_decoder(
            hidden_states=smiles_embed,
            attention_mask=attention_mask,
            encoder_hidden_states=admet_embed,
        )
        logits = self.to_logits(output['last_hidden_state'])
        return {'logits': logits}

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out

class MultiModalDecoder(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
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
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        # blocks
        hidden_states = self.drop(hidden_states)
        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states.contiguous(),
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states.contiguous(),
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

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
