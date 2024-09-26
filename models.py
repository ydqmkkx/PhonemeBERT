import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import ModelOutput
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.configuration_utils import PretrainedConfig
from vocab import _token


class PhonemeBertConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=len(_token)+20,
        embedding_size=768, # word embedding size
        hidden_size=768,
        output_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing = False,
        position_biased_input = False,
        position_embedding_type="relative_key_query",
        use_cache=False,
        
        grapheme_vocab_size = 84481,
        grapheme_max_position_embeddings=1024,

        use_sup_phoneme=False,
        sup_phoneme_vocab_size=30000,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_token_id=pad_token_id
        self.position_biased_input = position_biased_input
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

        self.grapheme_vocab_size = grapheme_vocab_size
        self.grapheme_max_position_embeddings = grapheme_max_position_embeddings

        self.use_sup_phoneme = use_sup_phoneme
        self.sup_phoneme_vocab_size = sup_phoneme_vocab_size


class PhonemeBertFFN(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.in_dense = nn.Linear(input_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.out_dense = nn.Linear(config.intermediate_size, input_size)
        self.LayerNorm = nn.LayerNorm(input_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor, position_embeddings: torch.Tensor=None) -> torch.Tensor:
        hidden_states = self.in_dense(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.out_dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if position_embeddings is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor + position_embeddings)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    

class PhonemeBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=config.pad_token_id)
        if config.use_sup_phoneme:
            self.sup_phoneme_embeddings = nn.Embedding(config.sup_phoneme_vocab_size, self.embedding_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        if not config.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sup_phoneme_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)
            
        embeddings = inputs_embeds

        if self.config.use_sup_phoneme:
            if sup_phoneme_ids is None:
                raise ValueError("Sup-phoneme should be input.")
            sup_phoneme_embeddings = self.sup_phoneme_embeddings(sup_phoneme_ids)
            embeddings += sup_phoneme_embeddings

        if self.config.position_biased_input:
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_input_ids(self, input_ids):
        # e.g. incremental_indices: [1, 2, 3, 0, 0]
        mask = input_ids.ne(self.config.pad_token_id).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + self.config.pad_token_id
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(
            self.config.pad_token_id + 1, sequence_length + self.config.pad_token_id + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
    

class PhonemeBertPreTrainedModel(PreTrainedModel):
    config_class = PhonemeBertConfig
    base_model_prefix = "bert"
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RobertaEncoder):
            module.gradient_checkpointing = value
    
    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        if not config.tie_word_embeddings:
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class PhonemeBertModel(PhonemeBertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Changed from transformers.models.roberta.modeling_roberta
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = PhonemeBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    # We remove the functions: get_input_embeddings, set_input_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sup_phoneme_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            raise ValueError("We do not expect the model to be decoder-based")

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if input_ids is None and attention_mask is None:
            raise ValueError("You have to input the atttention_mask when inputting inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = None

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()
        self.attention_mask = attention_mask

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            sup_phoneme_ids=sup_phoneme_ids,
            inputs_embeds=inputs_embeds,
        )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask, # None
            past_key_values=past_key_values, # None
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class PhonemeBertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.output_size, config.output_size)
        self.LayerNorm = nn.LayerNorm(config.output_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.decoder = nn.Linear(config.output_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class PhonemeBertSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.output_size, config.output_size)
        self.LayerNorm = nn.LayerNorm(config.output_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.decoder = nn.Linear(config.output_size, config.sup_phoneme_vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.sup_phoneme_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class PhonemeBertP2GHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.output_size, config.output_size)
        self.LayerNorm = nn.LayerNorm(config.output_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.decoder = nn.Linear(config.output_size, config.grapheme_vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.grapheme_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output
    

@dataclass
class PlbertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    p2g_loss: Optional[torch.FloatTensor] = None
    mlm_prediction_logits: torch.FloatTensor = None
    p2g_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PlbertForPreTraining(PhonemeBertPreTrainedModel):
    #_keys_to_ignore_on_save = [r"position_ids", r"mlm_head.decoder.weight", r"mlm_head.decoder.bias", r"p2g_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"mlm_head.decoder.weight", r"mlm_head.decoder.bias", r"p2g_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = PhonemeBertModel(config, add_pooling_layer=False)
        self.mlm_head = PhonemeBertMLMHead(config)
        self.p2g_head = PhonemeBertP2GHead(config)
        self.post_init()
        self._tie_or_clone_weights(self.mlm_head.decoder, self.bert.embeddings.word_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        tgt_ids: Optional[torch.LongTensor] = None,
        sup_phoneme_ids: Optional[torch.LongTensor] = None,
        sup_phoneme_labels: Optional[torch.LongTensor] = None,
        pooling_matrices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], PlbertForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sup_phoneme_ids=sup_phoneme_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        mlm_prediction_scores = self.mlm_head(sequence_output)
        p2g_prediction_scores = self.p2g_head(sequence_output)

        total_loss = None
        if labels is not None and tgt_ids is not None:
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            p2g_loss = loss_fct(p2g_prediction_scores.view(-1, self.config.grapheme_vocab_size), tgt_ids.view(-1))
            total_loss = mlm_loss + p2g_loss

        if not return_dict:
            output = (mlm_prediction_scores, p2g_prediction_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return PlbertForPreTrainingOutput(
            loss=total_loss,
            mlm_loss = mlm_loss,
            p2g_loss = p2g_loss,
            mlm_prediction_logits = mlm_prediction_scores,
            p2g_prediction_logits = p2g_prediction_scores,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )


@dataclass
class MpbertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    sp_loss: Optional[torch.FloatTensor] = None
    mlm_prediction_logits: torch.FloatTensor = None
    sp_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MpbertForPreTraining(PhonemeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"mlm_head.decoder.weight", r"mlm_head.decoder.bias", r"sp_head.decoder.weight",r"sp_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = PhonemeBertModel(config, add_pooling_layer=False)
        self.mlm_head = PhonemeBertMLMHead(config)
        self.sp_head = PhonemeBertSPHead(config)
        self.post_init()
        self._tie_or_clone_weights(self.mlm_head.decoder, self.bert.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.sp_head.decoder, self.bert.embeddings.sup_phoneme_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        tgt_ids: Optional[torch.LongTensor] = None,
        sup_phoneme_ids: Optional[torch.LongTensor] = None,
        sup_phoneme_labels: Optional[torch.LongTensor] = None,
        pooling_matrices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MpbertForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sup_phoneme_ids=sup_phoneme_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        mlm_prediction_scores = self.mlm_head(sequence_output)
        sp_prediction_scores = self.sp_head(sequence_output)
        pooling_matrices = pooling_matrices.to(sp_prediction_scores.dtype)
        sp_prediction_scores = torch.einsum('bse, bps -> bpe', sp_prediction_scores, pooling_matrices).contiguous()

        total_loss = None
        if labels is not None and tgt_ids is not None:
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sp_loss = loss_fct(sp_prediction_scores.view(-1, self.config.sup_phoneme_vocab_size), sup_phoneme_labels.view(-1))
            total_loss = mlm_loss + sp_loss

        if not return_dict:
            output = (mlm_prediction_scores, sp_prediction_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MpbertForPreTrainingOutput(
            loss=total_loss,
            mlm_loss = mlm_loss,
            sp_loss = sp_loss,
            mlm_prediction_logits = mlm_prediction_scores,
            sp_prediction_logits = sp_prediction_scores,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )