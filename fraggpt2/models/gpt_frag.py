import torch
import torch.nn as nn
from utils.utils import accuracy2
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from models import register_model
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from models.PocketTransformer import PocketTransformer3D
from transformers import AdamW, GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2Config
# from models.model_gpt2 import GPT2Model as myGPT2Model
from einops import rearrange, reduce, repeat
from torch import einsum

ESM = 1280
OUTFEATURES = 768
ESM_LEN_OUT = 40
ESM_LEN_IN = 200

mapping = {'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
           'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
           'Y': 19, 'V': 20}


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class MultiHeadCrossAttentionWithMask(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadCrossAttentionWithMask, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm = LayerNorm(hidden_dim)
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_length = query.shape[1]

        query = self.norm(query)
        key = self.norm(key)
        value = self.norm(value)
        # Project inputs to hidden_dim
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Split into heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_probs, value)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        # Final linear layer
        context = self.output_linear(context)

        return context, attention_probs


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

    def forward(self, x, context, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # ----pre-layernorm, for queries and context--------

        x = self.norm(x)
        context = self.context_norm(context)
        # get queries
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        # scale
        q = q * self.scale
        # get key / values
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # query / key similarity
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        if mask is not None:
            # Apply mask to attention similarity scores
            sim = sim.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
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


class ESM2_encoder(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        # self.padding_idx = mapping['PAD']

        self.linear_layer_for_all = nn.Linear(in_feature, out_feature)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.linear_layer_for_all.bias, 0)
        nn.init.constant_(self.linear_layer_for_all.weight, 0)

    def forward(self, x):
        # 1280 768
        x = self.linear_layer_for_all(x)

        return x


class ESM2_encoder_prefix(nn.Module):
    def __init__(self, in_feature, out_feature, esm_len_in, esm_len_out):
        super().__init__()
        # self.padding_idx = mapping['PAD']

        self.linear_layer_for_all = nn.Linear(in_feature, out_feature)
        self.linear_layer_for_len = nn.Linear(esm_len_in, esm_len_out)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.linear_layer_for_all.bias, 0)
        nn.init.constant_(self.linear_layer_for_all.weight, 0)
        nn.init.constant_(self.linear_layer_for_len.bias, 0)
        nn.init.constant_(self.linear_layer_for_len.weight, 0)

    def forward(self, x):
        # 8 512 768   8 768
        # print(x.shape)
        x = self.linear_layer_for_all(x)
        x = rearrange(x, 'b n c -> b c n')
        x = self.linear_layer_for_len(x)
        x = rearrange(x, 'b n c -> b c n')
        return x


@register_model(['smiles_frag', 'smiles_frag_admet','fine_tune','FragGPT_admet'])
class FragSmilesGPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN
        )
        super().__init__(config)

        self.cfg = cfg


        if cfg.MODEL.ADMET_ENCODER.use_admet:
            # self.admet_encoder_proj = nn.Linear(1, cfg.MODEL.GPT_MODEL.n_embd)
            self.admet_encoder_proj = nn.Linear(cfg.MODEL.GPT_MODEL.n_embd,
                                                cfg.MODEL.GPT_MODEL.n_embd)
            admet_config = BertConfig(
                hidden_size=cfg.MODEL.ADMET_ENCODER.n_embd,
                num_attention_heads=cfg.MODEL.ADMET_ENCODER.n_head,
                num_hidden_layers=cfg.MODEL.ADMET_ENCODER.n_layer,
            )
            self.admet_encoder = BertModel(admet_config)

        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            admet: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if admet is not None and self.cfg.MODEL.ADMET_ENCODER.use_admet:
            admet_prop = admet.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.GPT_MODEL.n_embd).type(admet.dtype)
            admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
            admet_attention_mask = torch.ones((admet_prop.shape[0], admet_prop.shape[1]), device=admet_prop.device)
            admet_hidden = self.admet_encoder(
                inputs_embeds=admet_prop_inputs_embd,
                attention_mask=admet_attention_mask,
            ).last_hidden_state

            # admet_prop = admet.unsqueeze(-1)
            # admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
            # admet_hidden = admet_prop_inputs_embd

            inputs_embeds = self.transformer.wte(input_ids)
            inputs_embeds = torch.cat([admet_hidden, inputs_embeds[:, 1:]], dim=1)
            res_len = 0
            if inputs_embeds.shape[1] > self.cfg.DATA.MAX_SMILES_LEN:
                res_len = inputs_embeds.shape[1] - self.cfg.DATA.MAX_SMILES_LEN
            inputs_embeds = inputs_embeds[:, :self.cfg.DATA.MAX_SMILES_LEN]
            transformer_outputs = self.transformer(
                # input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,  # encoder_rep
                encoder_attention_mask=encoder_attention_mask,  # encoder_attention_mask
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., 39:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if res_len > 0:
                    shift_labels = labels[..., 1:-res_len].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if res_len > 0:
                    acc = accuracy2(lm_logits[:, 39:-1], labels[:, 1:-res_len])
                else:
                    acc = accuracy2(lm_logits[:, 39:-1], labels[:, 1:])
        else:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )


@register_model(['smiles_frag_pocket', 'smiles_frag_pocket_pretrain'])
class FragSmilesPocketGPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        if cfg.MODEL.ESM.use_esm is False:
            cfg.MODEL.ESM.cross_attention = True
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN,
            add_cross_attention=cfg.MODEL.ESM.cross_attention,
        )
        super().__init__(config)
        self.cfg = cfg
        if cfg.MODEL.ESM.use_esm and cfg.MODEL.ESM.q_fromer:
            self.esm_queries = nn.Parameter(torch.randn(50, cfg.MODEL.ESM.input_dim))
            self.esm_q_former = CrossAttention(
                dim=cfg.MODEL.ESM.input_dim,
                context_dim=cfg.MODEL.ESM.input_dim,
                # context_dim=1280,
                heads=cfg.MODEL.ESM.num_heads,
                parallel_ff=True
            )
            self.admet_attn_pool_norm = LayerNorm(cfg.MODEL.ESM.input_dim)
        # -------------------esm------------------------
        if cfg.MODEL.ESM.use_esm :
            if cfg.MODEL.ESM.prefix:
                self.esm2_encoder = ESM2_encoder_prefix(ESM, OUTFEATURES, ESM_LEN_IN, ESM_LEN_OUT)
            else:
                self.esm2_encoder = ESM2_encoder(ESM, OUTFEATURES)
        if cfg.MODEL.ADMET_ENCODER.use_admet:
            self.admet_encoder_proj = nn.Linear(cfg.MODEL.GPT_MODEL.n_embd,
                                                cfg.MODEL.GPT_MODEL.n_embd)
            admet_config = BertConfig(
                hidden_size=cfg.MODEL.ADMET_ENCODER.n_embd,
                num_attention_heads=cfg.MODEL.ADMET_ENCODER.n_head,
                num_hidden_layers=cfg.MODEL.ADMET_ENCODER.n_layer,
            )
            self.admet_encoder = BertModel(admet_config)

        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.padding_idx = mapping['PAD']
        self.encoder = PocketTransformer3D(**cfg.MODEL.POCKET_ENCODER)
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pocket_seq: Optional[torch.LongTensor] = None,
            pocket_edge_type: Optional[torch.LongTensor] = None,
            pocket_dis: Optional[torch.FloatTensor] = None,
            admet: Optional[torch.FloatTensor] = None,
            padding_mask: Optional[torch.FloatTensor] = None,
            esm2: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        if self.cfg.MODEL.ESM.use_esm and esm2 is not None and self.cfg.MODEL.ESM.cross_attention:
            encoder_rep = self.esm2_encoder(esm2)
            encoder_attention_mask = 1 - padding_mask.type(torch.int64)
            encoder_attention_mask = encoder_attention_mask[:, :, 1]
        elif self.cfg.MODEL.ESM.use_esm is False:

            encoder_rep, padding_mask_ = self.encoder(src_tokens=pocket_seq,
                                                      src_distance=pocket_dis,
                                                      src_edge_type=pocket_edge_type)
            encoder_attention_mask = 1 - padding_mask_.type(torch.int64)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if admet is not None and self.cfg.MODEL.ADMET_ENCODER.use_admet:
            admet_prop = admet.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.GPT_MODEL.n_embd).type(
                self.admet_encoder_proj.weight.dtype)
            admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
            admet_attention_mask = torch.ones((admet_prop.shape[0], admet_prop.shape[1]), device=admet_prop.device)
            admet_hidden = self.admet_encoder(
                inputs_embeds=admet_prop_inputs_embd,
                attention_mask=admet_attention_mask,
            ).last_hidden_state

            inputs_embeds = self.transformer.wte(input_ids)
            inputs_embeds = torch.cat([admet_hidden, inputs_embeds[:, 1:]], dim=1)
            res_len = 0
            if inputs_embeds.shape[1] > self.cfg.DATA.MAX_SMILES_LEN:
                res_len = inputs_embeds.shape[1] - self.cfg.DATA.MAX_SMILES_LEN
            inputs_embeds = inputs_embeds[:, :self.cfg.DATA.MAX_SMILES_LEN]

            transformer_outputs = self.transformer(
                # input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_rep,  # encoder_rep
                encoder_attention_mask=encoder_attention_mask,  # encoder_attention_mask
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., 39:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if res_len > 0:
                    shift_labels = labels[..., 1:-res_len].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if res_len > 0:
                    acc = accuracy2(lm_logits[:, 39:-1], labels[:, 1:-res_len])
                else:
                    acc = accuracy2(lm_logits[:, 39:-1], labels[:, 1:])
        elif self.cfg.MODEL.ESM.use_esm and esm2 is not None and self.cfg.MODEL.ESM.cross_attention is False:
            inputs_embeds = self.transformer.wte(input_ids)
            prefix = None
            if self.cfg.MODEL.ESM.q_fromer and self.cfg.MODEL.ESM.prefix:
                raise ('q_fromer and prefix can not be true at the same time')

            if self.cfg.MODEL.ESM.q_fromer:
                esm = self.esm2_encoder(esm2) # n 200 768
                bs = esm.shape[0]
                q = self.esm_queries.unsqueeze(0).repeat(bs, 1, 1)
                mask = padding_mask[:, :, 1].type(torch.int)
                context = self.esm_q_former(q, esm, mask)
                prefix = context
            elif self.cfg.MODEL.ESM.prefix:
                prefix = self.esm2_encoder(esm2)
            if prefix is not None:
                inputs_embeds = torch.cat([prefix, inputs_embeds[:, 1:]], dim=1)
            else:
                inputs_embeds = inputs_embeds

            res_len = 0
            if inputs_embeds.shape[1] > self.cfg.DATA.MAX_SMILES_LEN:
                res_len = inputs_embeds.shape[1] - self.cfg.DATA.MAX_SMILES_LEN
            inputs_embeds = inputs_embeds[:, : self.cfg.DATA.MAX_SMILES_LEN, :]
            prefix_len = prefix.shape[1]
            transformer_outputs = self.transformer(
                # input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,  # encoder_rep
                encoder_attention_mask=encoder_attention_mask,  # encoder_attention_mask
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)
            lm_logits = self.lm_head(hidden_states)  # 16 512
            loss = None
            if labels is not None:  # 16 374
                shift_logits = lm_logits[..., prefix_len - 1:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if res_len > 0:
                    shift_labels = labels[..., 1:-res_len].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if res_len > 0:
                    acc = accuracy2(lm_logits[:, prefix_len - 1:-1], labels[:, 1:-res_len])
                else:
                    acc = accuracy2(lm_logits[:, prefix_len - 1:-1], labels[:, 1:])
        else:
            transformer_outputs = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_rep,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )


@register_model(['target'])
class FragSmilesTargetGPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN
        )
        super().__init__(config)
        self.cfg = cfg
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

@register_model(['target_living2'])
class FragSmiles_target_living2GPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN,
            add_cross_attention=True,
        )
        super().__init__(config)
        self.cfg = cfg

        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.encoder = PocketTransformer3D(**cfg.MODEL.POCKET_ENCODER)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pocket_seq: Optional[torch.LongTensor] = None,
        pocket_edge_type: Optional[torch.LongTensor] = None,
        pocket_dis: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        encoder_rep, padding_mask = self.encoder(src_tokens=pocket_seq,
                                                 src_distance=pocket_dis,
                                                 src_edge_type=pocket_edge_type)

        encoder_attention_mask = 1 - padding_mask.type(torch.int64)

        # encoder_rep, encoder_attention_mask = None, None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_rep,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )


@register_model(['target_living'])
class FragSmilesPocketLivingGPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN,
            add_cross_attention=cfg.MODEL.ESM.cross_attention,
        )
        super().__init__(config)
        self.cfg = cfg
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # -------------------esm ------------------------
        self.esm2_encoder = ESM2_encoder_prefix(ESM, OUTFEATURES, ESM_LEN_IN, ESM_LEN_OUT)
        self.padding_idx = mapping['PAD']
        self.encoder = PocketTransformer3D(**cfg.MODEL.POCKET_ENCODER)
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            padding_mask: Optional[torch.FloatTensor] = None,
            # --------------------------
            esm2: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        encoder_rep = self.esm2_encoder(esm2)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.transformer.wte(input_ids)
        prefix = torch.randn(encoder_rep.shape).type(torch.float32).cuda()
        # prefix = torch.ones_like(encoder_rep).type(torch.float32).cuda()
        if self.cfg.MODEL.ESM.gen_add_pocket:
            inputs_embeds = torch.cat([encoder_rep, inputs_embeds[:, 1:]], dim=1)
        else:
            inputs_embeds = torch.cat([prefix, inputs_embeds[:, 1:]], dim=1)
        res_len = 0
        if inputs_embeds.shape[1] > self.cfg.DATA.MAX_SMILES_LEN:
            res_len = inputs_embeds.shape[1] - self.cfg.DATA.MAX_SMILES_LEN

        inputs_embeds = inputs_embeds[:, :self.cfg.DATA.MAX_SMILES_LEN]
        if encoder_rep is not None:
            prefix_len = encoder_rep.shape[1] - res_len
        else:
            prefix_len = 0
        transformer_outputs = self.transformer(
            # input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,  # encoder_rep
            encoder_attention_mask=encoder_attention_mask,  # encoder_attention_mask
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # hidden_states = transformer_outputs[prefix_len:, :]
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., prefix_len - 1:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if res_len > 0:
                shift_labels = labels[..., 1:-res_len].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if res_len > 0:
                acc = accuracy2(lm_logits[:, prefix_len - 1:-1], labels[:, 1:-res_len])
            else:
                acc = accuracy2(lm_logits[:, prefix_len - 1:-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
