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

@register_model(['smiles_admet'])
class SmiLES_ADMET(nn.Module):
    def __init__(self, cfg, task, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.caption_loss_weight = cfg.MODEL.CAPTION_LOSS_WEIGHT
        self.contrastive_loss_weight = cfg.MODEL.CONTRASTIVE_LOSS_WEIGHT
        self.matching_loss_weight = cfg.MODEL.MATCHING_LOSS_WEIGHT
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
        self.smiles_proj = nn.Linear(cfg.MODEL.UNIMODAL.n_embd, cfg.MODEL.UNIMODAL.n_embd)

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

        self.admet_frag_proj = nn.Linear(cfg.MODEL.ADMET_MODEL.n_embd, cfg.MODEL.UNIMODAL.n_embd)

        # query transformer & attn pool  for admet
        # self.admet_queries = nn.Parameter(torch.randn(54 + 1, cfg.MODEL.UNIMODAL.n_embd))
        # self.admet_attn_pool = CrossAttention(
        #     dim=cfg.MODEL.UNIMODAL.n_embd,
        #     context_dim=cfg.MODEL.ADMET_MODEL.n_embd,
        #     dim_head=cfg.MODEL.ADMET_MODEL.n_embd//cfg.MODEL.ADMET_MODEL.n_head,
        #     heads=cfg.MODEL.ADMET_MODEL.n_head,
        #     norm_context=True
        # )
        # self.admet_attn_pool_norm = LayerNorm(cfg.MODEL.UNIMODAL.n_embd)
        self.type_embed = nn.Embedding(3, cfg.MODEL.ADMET_MODEL.n_embd)

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
        # self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.MODEL.TEMPERATURE), requires_grad=True)
        self.alpha = cfg.MODEL.ALPHA
        self.asm_head = nn.Linear(cfg.MODEL.MULTIMODAL.n_embd, 2)

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(self,
        smiles_ids: torch.LongTensor,
        admet_prop: torch.tensor,
        labels: torch.LongTensor = None,
    ):
        bs = smiles_ids.shape[0]
        # unimodal decoding
        smiles_cls_proj_embed, smiles_embed, attention_mask = self.embed_smiles(smiles_ids=smiles_ids)

        # admet encoding
        admet_cls_token_feature = self.admet_cls_token.weight.unsqueeze(0).repeat(bs, 1, 1)
        admet_prop = admet_prop.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.ADMET_MODEL.n_embd).type(smiles_cls_proj_embed.dtype)
        admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
        admet_prop_frag_inputs_embd = admet_prop_inputs_embd
        type_ids = torch.LongTensor([0 for _ in range(len(classification_names)-len(drop_names))] +
                                    [1 for _ in range(len(regression_names))]).to(admet_cls_token_feature.device)
        type_embed = self.type_embed(type_ids).type(admet_cls_token_feature.dtype)
        admet_prop_frag_inputs_embd = admet_prop_frag_inputs_embd + type_embed
        admet_prop_frag_inputs_embeds = torch.cat([admet_cls_token_feature, admet_prop_frag_inputs_embd], dim=1)
        admet_frag_attention_mask = torch.ones((bs, len(tasks)-len(drop_names)+1), device=admet_cls_token_feature.device)

        admet_hidden = self.admet_encoder(
            inputs_embeds=admet_prop_frag_inputs_embeds,
            attention_mask=admet_frag_attention_mask,
        ).last_hidden_state

        # admet attention pooling
        admet_cls_embed, admet_embed = admet_hidden[:, 0], admet_hidden
        admet_cls_proj_embed = self.admet_frag_proj(admet_cls_embed)
        # multimodal decoding
        output = self.multimodal_decoder(
            hidden_states         = smiles_embed,
            attention_mask        = attention_mask,
            encoder_hidden_states = admet_embed,
            encoder_attention_mask= admet_frag_attention_mask,
        )
        logits = self.to_logits(output['last_hidden_state'][:, :-1])

        # contrast loss
        smiles_cls_proj_embed = F.normalize(smiles_cls_proj_embed, dim=-1)
        admet_cls_proj_embed = F.normalize(admet_cls_proj_embed, dim=-1)

        logit_scale = self.logit_scale.exp()

        sim_a2s = logit_scale * admet_cls_proj_embed @ smiles_cls_proj_embed.t()
        sim_s2a = logit_scale * smiles_cls_proj_embed @ admet_cls_proj_embed.t()

        sim_targets = torch.zeros(sim_a2s.size()).to(sim_a2s.device)
        sim_targets.fill_diagonal_(1)
        # soft label
        sim_a2s_targets = self.alpha * F.softmax(sim_a2s, dim=1) + (1 - self.alpha) * sim_targets
        sim_s2a_targets = self.alpha * F.softmax(sim_s2a, dim=1) + (1 - self.alpha) * sim_targets

        loss_a2s = -torch.sum(F.log_softmax(sim_a2s, dim=1) * sim_a2s_targets, dim=1).mean()
        loss_s2a = -torch.sum(F.log_softmax(sim_s2a, dim=1) * sim_s2a_targets, dim=1).mean()
        loss_asa = (loss_a2s + loss_s2a) / 2

        # matching loss
        with torch.no_grad():
            weights_a2s = F.softmax(sim_a2s[:, :bs], dim=1)
            weights_s2a = F.softmax(sim_s2a[:, :bs], dim=1)

            weights_a2s.fill_diagonal_(0)
            weights_s2a.fill_diagonal_(0)

        # select a negative graph for each smi
        smiles_embeds_neg = []
        smiles_atts_neg = []
        for b in range(bs):
            if float(torch.sum(weights_a2s[b])) <= 0:
                weights_a2s[b] = torch.rand(1, bs).type(weights_a2s.type()).to(weights_a2s)
                weights_a2s[b, b] = 0.0
            try:
                neg_idx = torch.multinomial(weights_a2s[b], 1).item()
            except:
                weights_a2s[b] = torch.ones(1, bs).type(weights_a2s.type()).to(weights_a2s)
                weights_a2s[b, b] = 0.00000001
                neg_idx = torch.multinomial(weights_a2s[b], 1).item()
            smiles_embeds_neg.append(smiles_embed[neg_idx])
            smiles_atts_neg.append(attention_mask[neg_idx])
        smiles_embeds_neg = torch.stack(smiles_embeds_neg, dim=0)
        smiles_atts_neg = torch.stack(smiles_atts_neg, dim=0)

        # select a negative smi for each graph
        admet_embeds_neg = []
        admet_atts_neg = []
        for b in range(bs):
            if float(torch.sum(weights_s2a[b])) <= 0:
                weights_s2a[b] = torch.rand(1, bs).type(weights_s2a.type()).to(weights_s2a)
                weights_s2a[b, b] = 0.0
            try:
                neg_idx = torch.multinomial(weights_s2a[b], 1).item()
            except:
                weights_s2a[b] = torch.ones(1, bs).type(weights_s2a.type()).to(weights_s2a)
                weights_s2a[b, b] = 0.00000001
                neg_idx = torch.multinomial(weights_s2a[b], 1).item()
            admet_embeds_neg.append(admet_embed[neg_idx])
            admet_atts_neg.append(admet_frag_attention_mask[neg_idx])
        admet_embeds_neg = torch.stack(admet_embeds_neg, dim=0)
        admet_atts_neg = torch.stack(admet_atts_neg, dim=0)

        smiles_embeds_all = torch.cat([smiles_embed, smiles_embeds_neg], dim=0)
        smiles_atts_all = torch.cat([attention_mask, smiles_atts_neg], dim=0)

        smi_embeds_all = torch.cat([admet_embeds_neg, admet_embed], dim=0)
        smi_atts_all = torch.cat([admet_atts_neg, admet_frag_attention_mask], dim=0)

        output_neg = self.multimodal_decoder(
            hidden_states=smiles_embeds_all,
            attention_mask=smiles_atts_all,
            encoder_hidden_states=smi_embeds_all,
            encoder_attention_mask=smi_atts_all,
        )['last_hidden_state']

        vl_embeddings = torch.cat([output['last_hidden_state'][:, -1, :], output_neg[:, -1, :]], dim=0)
        vl_output = self.asm_head(vl_embeddings)
        asm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            vl_output.device)
        loss_matching = F.cross_entropy(vl_output, asm_labels)

        acc = accuracy2(logits[:, :-1], labels[:, 1:])

        # caption loss
        caption_loss = self.calc_caption_loss(logits, labels)

        loss = self.caption_loss_weight * caption_loss + \
               self.contrastive_loss_weight * loss_asa + \
               self.matching_loss_weight * loss_matching

        return {
            'loss': loss,
            'caption_loss': caption_loss,
            'contrastive_loss': loss_asa,
            'matching_loss': loss_matching,
            'hit@1': acc,
        }


    def embed_smiles(self, smiles_ids):
        inputs_embeds = self.unimodal_decoder.wte(smiles_ids)
        attention_mask = smiles_ids.ne(self.frag_padding_idx).type(torch.int64)

        unimodal_output = self.unimodal_decoder(
            inputs_embeds =inputs_embeds,
            attention_mask=attention_mask,
        )

        cls_embed = unimodal_output['last_hidden_state'][:, -1]
        smiles_embed = unimodal_output['last_hidden_state']

        # get text cls token
        cls_proj_embed = self.smiles_proj(cls_embed)
        return cls_proj_embed, smiles_embed, attention_mask


    def calc_caption_loss(self, lm_logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        caption_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        caption_loss = caption_loss * self.caption_loss_weight
        return caption_loss

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
        frag_attention_mask_1d = fragment_ids.ne(self.frag_padding_idx).type(torch.int64)

        frag_attention_mask = frag_attention_mask_1d[:, None, None, :]
        frag_attention_mask = frag_attention_mask.to(dtype=frag_inputs_embeds.dtype)  # fp16 compatibility
        frag_attention_mask = (1.0 - frag_attention_mask) * torch.finfo(frag_inputs_embeds.dtype).min

        frag_hidden = self.frag_encoder(
            hidden_states=frag_inputs_embeds,
            attention_mask=frag_attention_mask,
        ).last_hidden_state

        # unimodal decoding
        # smiles_cls_proj_embed, smiles_embed, attention_mask = self.embed_smiles(smiles_ids=smiles_ids)
        inputs_embeds = self.unimodal_decoder.wte(smiles_ids)
        attention_mask = smiles_ids.ne(self.frag_padding_idx).type(torch.int64)

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
        admet_prop_frag_inputs_embd = torch.cat([admet_prop_inputs_embd, frag_hidden], dim=1)
        type_ids = torch.LongTensor([0 for _ in range(len(classification_names) - len(drop_names))] +
                                    [1 for _ in range(len(regression_names))] +
                                    [2 for _ in range(frag_hidden.shape[1])]).to(admet_cls_token_feature.device)
        type_embed = self.type_embed(type_ids).type(admet_cls_token_feature.dtype)
        admet_prop_frag_inputs_embd = admet_prop_frag_inputs_embd + type_embed
        admet_prop_frag_inputs_embeds = torch.cat([admet_cls_token_feature, admet_prop_frag_inputs_embd], dim=1)
        admet_frag_attention_mask = torch.cat(
            [torch.ones((bs, len(tasks) - len(drop_names) + 1), device=admet_cls_token_feature.device),
             frag_attention_mask_1d], dim=1)
        admet_hidden = self.admet_encoder(
            inputs_embeds=admet_prop_frag_inputs_embeds,
            attention_mask=admet_frag_attention_mask,
        ).last_hidden_state

        admet_embed = admet_hidden
        # multimodal decoding
        output = self.multimodal_decoder(
            hidden_states=smiles_embed,
            attention_mask=attention_mask,
            encoder_hidden_states=admet_embed,
            encoder_attention_mask=admet_frag_attention_mask,
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
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        if encoder_attention_mask is None:
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
