import torch
import torch.nn as nn
from utils.utils import accuracy2
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from models import register_model

from transformers import BertConfig, BertModel
from models.PocketTransformer import PocketTransformer3D
from models.mygpt import MyGPT2Model
from transformers import AdamW, GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2Config

@register_model(['smiles_frag', 'smiles_frag_admet'])
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
        if self.cfg.MODEL.ADMET_ENCODER.use_admet:
            self.transformer = MyGPT2Model(config)
        else:
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
                self_attn_length=admet_hidden.shape[1],
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
            ss = admet_hidden.shape[1] - 1
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., ss:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if res_len > 0:
                    shift_labels = labels[..., 1:-res_len].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if res_len > 0:
                    acc = accuracy2(lm_logits[:, ss:-1], labels[:, 1:-res_len])
                else:
                    acc = accuracy2(lm_logits[:, ss:-1], labels[:, 1:])
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

@register_model(['smiles_frag_pocket', 'smiles_frag_pocket_pretrain', 'target'])
class FragSmilesPocketGPT(GPT2LMHeadModel):
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
        if cfg.MODEL.ADMET_ENCODER.use_admet:
            self.admet_encoder_proj = nn.Linear(cfg.MODEL.GPT_MODEL.n_embd,
                                                cfg.MODEL.GPT_MODEL.n_embd)
            admet_config = BertConfig(
                hidden_size=cfg.MODEL.ADMET_ENCODER.n_embd,
                num_attention_heads=cfg.MODEL.ADMET_ENCODER.n_head,
                num_hidden_layers=cfg.MODEL.ADMET_ENCODER.n_layer,
            )
            self.admet_encoder = BertModel(admet_config)
            # self.type_embed = nn.Embedding(2, cfg.MODEL.ADMET_ENCODER.n_embd)
            # fusion_config = BertConfig(
            #     hidden_size=cfg.MODEL.ADMET_ENCODER.n_embd,
            #     num_attention_heads=cfg.MODEL.ADMET_ENCODER.n_head,
            #     num_hidden_layers=1,
            # )
            # self.fusion_encoder = BertModel(fusion_config)

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
        encoder_rep, padding_mask = self.encoder(src_tokens=pocket_seq,
                                                 src_distance=pocket_dis,
                                                 src_edge_type=pocket_edge_type)

        encoder_attention_mask = 1 - padding_mask.type(torch.int64)

        # encoder_rep, encoder_attention_mask = None, None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # admet & cross attention
        # if admet is not None and self.cfg.MODEL.ADMET_ENCODER.use_admet:
        #     admet_prop = admet.unsqueeze(-1).repeat(1, 1, encoder_rep.shape[-1]).type(encoder_rep.dtype)
        #     admet_prop_inputs_embd = self.admet_encoder_proj(admet_prop)
        #     admet_attention_mask = torch.ones((admet_prop.shape[0], admet_prop.shape[1]), device=admet_prop.device)
        #     admet_hidden = self.admet_encoder(
        #         inputs_embeds=admet_prop_inputs_embd,
        #         attention_mask=admet_attention_mask,
        #     ).last_hidden_state
        #     assert encoder_rep.shape[-1] == admet_hidden.shape[-1]
        #     new_encoder_rep = torch.cat([admet_hidden, encoder_rep], dim=1)
        #     type_ids = torch.LongTensor([0 for _ in range(admet_hidden.shape[1])] +
        #                                 [1 for _ in range(encoder_rep.shape[1])]).to(encoder_rep.device)
        #     new_encoder_rep = new_encoder_rep + self.type_embed(type_ids).type(encoder_rep.dtype).unsqueeze(0)
        #
        #     encoder_attention_mask = torch.cat([admet_attention_mask, encoder_attention_mask], dim=1)
        #     new_encoder_rep = self.fusion_encoder(
        #         inputs_embeds=new_encoder_rep,
        #         attention_mask=encoder_attention_mask,
        #     ).last_hidden_state
        #     encoder_rep = new_encoder_rep

        # admet & prefix
        if admet is not None and self.cfg.MODEL.ADMET_ENCODER.use_admet:
            admet_prop = admet.unsqueeze(-1).repeat(1, 1, self.cfg.MODEL.GPT_MODEL.n_embd).type(self.admet_encoder_proj.weight.dtype)
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
                encoder_hidden_states=encoder_rep, # encoder_rep
                encoder_attention_mask=encoder_attention_mask, # encoder_attention_mask
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
