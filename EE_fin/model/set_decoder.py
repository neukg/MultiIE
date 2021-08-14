import torch.nn as nn
import torch
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertLayerNorm, BertSelfAttention


class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_events, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_events
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_events, config.hidden_size)
        

        self.event_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.event_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder2event = nn.Linear(config.hidden_size, num_classes * 2)

        
        torch.nn.init.orthogonal_(self.event_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.event_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)



    def forward(self, encoder_hidden_states, encoder_attention_mask):
        '''
        encoder_hidden_states: bsz, sl, h
        '''
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            # bsz, m, h
            hidden_states = layer_outputs[0]

        # bsz, sl, m, 2*r
        event_logits = self.decoder2event(torch.tanh(
            self.event_metric_1(hidden_states).unsqueeze(2) + self.event_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze().transpose(1, 2)

        return event_logits


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs