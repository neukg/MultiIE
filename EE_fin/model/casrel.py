# -*- encoding: utf-8 -*-

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from transformers import BertModel

import config
from .set_decoder import SetDecoder as EventDecoder


class BiMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.BCELoss(reduction='none')
    
    def forward(self, outputs:torch.tensor, targets:torch.tensor, attention_mask:torch.tensor):
        
        #ouputs: bsz, sl, event_num, class_num, 2
        #targets: bsz, sl, event_num, class_num, 2
        #attention_mask: bsz, sl
        
        batch_size, seq_len, event_num, class_num, _ = outputs.shape
        
        origin_outputs = outputs.transpose(1, 2)
        # bsz, 1, event_num, sl, class_num, 2
        outputs = origin_outputs.unsqueeze(1)
        
        origin_targets = targets.transpose(1, 2)
        # bsz, event_num, 1, sl, class_num, 2
        targets = origin_targets.unsqueeze(2)

        # bsz, event_num, event_num , sl, class_num, 2
        cost = outputs * targets
        # bsz, event_num, event_num
        cost = - torch.sum(cost, dim=[3, 4, 5])
        
        match_indice = np.array([linear_sum_assignment(c) for c in cost.detach().cpu().numpy()])

        # bsz , event_num
        target_indice = torch.from_numpy(match_indice[:, 0, :]).reshape(-1)
        output_indice = torch.from_numpy(match_indice[:, 1, :]).reshape(-1)
        
        batch_indice = torch.tensor([[i]*event_num for i in range(batch_size)]).reshape(-1)

        # bsz, sl, event_num(order), class_num, 2 
        new_target = origin_targets[batch_indice, target_indice, :, :, :].reshape(batch_size, event_num, seq_len, class_num, 2).transpose(1, 2)
        new_output = origin_outputs[batch_indice, output_indice, :, :, :].reshape(batch_size, event_num, seq_len, class_num, 2).transpose(1, 2)

        new_cost = self.loss_fct(new_output, new_target)
        new_cost = torch.sum(new_cost.mean(4), dim=[2, 3])
        attention_mask = attention_mask
        new_cost = torch.sum(new_cost * attention_mask.float()) / torch.sum(attention_mask.float())

        return new_cost
'''
class BiMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.BCELoss(reduction='none')
    
    def forward(self, outputs:torch.tensor, targets:torch.tensor, true_event_num:torch.tensor):
        
        #ouputs: bsz, event_num, label_num
        #targets: bsz, event_num, label_num
        
   
        origin_outputs = outputs
        origin_targets = targets

        # bsz, target_event_num, 1, label_num
        targets = origin_targets.unsqueeze(2)
        # bsz, 1 , event_num, label_num
        outputs = origin_outputs.unsqueeze(1)

        new_targets, new_outputs = [], []
        for i, e_num in enumerate(true_event_num.tolist()):
            if int(e_num) == 0:
                continue
            # target_event_num, event_num, label_num
            try:
                cost = outputs[i] * targets[i, 0 : int(e_num)]
            except:
                raise ValueError(i, e_num, targets.shape, outputs.shape)
            # target_event_num, event_num
            cost = - torch.sum(cost, dim=[2])
        
            match_indice = np.array(linear_sum_assignment(cost.detach().cpu().numpy()))

            # event_num
            target_indice = torch.from_numpy(match_indice[0, :]).reshape(-1)
            output_indice = torch.from_numpy(match_indice[1, :]).reshape(-1)
        
            # event_num(order), label_num
            new_targets.append(origin_targets[i, target_indice, :])
            new_outputs.append(origin_outputs[i, output_indice, :])
        # bsz*event_num, label_num
        try:
            new_targets = torch.cat(new_targets)
            new_outputs = torch.cat(new_outputs)
        except:
            raise ValueError(new_targets)
        new_cost = self.loss_fct(new_outputs, new_targets)
        new_cost = torch.sum(new_cost)

        return new_cost
'''

class ERENet(nn.Module):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, encoder, classes_num, event_num):
        super().__init__()
        self.classes_num = classes_num
        self.event_num = event_num
        self.encoder = encoder
        self.event_decoder = EventDecoder(encoder.config, self.event_num, config.common["decoder_layer"], self.classes_num, return_intermediate=False)
        
        hidden_size = encoder.config.hidden_size
        layer_num = encoder.config.num_hidden_layers
        # pointer net work
        self.subject_dense = nn.Linear(hidden_size, 2)

        # additive attention
        self.w1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w2 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.tanh = nn.Tanh()
        self.attn1 = nn.Linear(hidden_size // 2, 1)
        self.w3 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.context_dense = nn.Linear(hidden_size * 2, hidden_size)
        # BERT动态融合
        # trigger
        #self.trigger_weight = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layer_num)])
        # argument 
        #self.argument_weight = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layer_num)])

        self.loss_fct = nn.BCELoss(reduction='none')
        self.bi_match_loss = BiMatchLoss()

    def additive_attention(self, h, g, mask=None):
        x = self.attn1(self.tanh(self.w1(h) + self.w2(g)))
        if mask != None:
            x = x * mask.unsqueeze(2)
        x = x.squeeze(-1)
        weight = F.softmax(x, 1)
        return weight

    def forward(self, q_ids=None, passage_ids=None, segment_ids=None, attention_mask=None, subject_labels=None, object_labels=None, 
                event_num_labels=None,
                eval_file=None,
                is_eval=False):

        # bert_encoder(batch_size, seq_len, hidden_size)
        bert_encoder, pooler_out, hidden_states = self.encoder(passage_ids, attention_mask, segment_ids, output_hidden_states=True)
        # 1: trigger: last two  argument: last one
        trigger_represent = hidden_states[-2]
        argument_represent = bert_encoder
        # 2. trigger: dynamic fusion  argument: dynamic fusion
        #trigger_represent = self.dynamic_fusion(hidden_states=hidden_states[1:], weights=self.trigger_weight)
        #argument_represent = self.dynamic_fusion(hidden_states=hidden_states[1:], weights=self.argument_weight) 
        if not is_eval:
            seq_len = trigger_represent.size(1)
            start_sub = trigger_represent * subject_labels[:, :, 0].unsqueeze(2)
            start_sub = nn.AvgPool1d(seq_len)(start_sub.transpose(1, 2))
            start_sub = start_sub.transpose(1, 2)
            end_sub = trigger_represent * subject_labels[:, :, 1].unsqueeze(2)
            end_sub = nn.AvgPool1d(seq_len)(end_sub.transpose(1, 2))
            end_sub = end_sub.transpose(1, 2)

            # batch, 1, hidden_size * 2
            subject = torch.cat([start_sub, end_sub], 2)
            att_weight = self.additive_attention(argument_represent, subject, attention_mask).unsqueeze(2)
            context_encoder = argument_represent * att_weight
            context_encoder = torch.cat([context_encoder, argument_represent], dim=2)
            context_encoder = self.context_dense(context_encoder)

            sub_preds = nn.Sigmoid()(self.subject_dense(trigger_represent))
            po_logits = self.event_decoder(context_encoder, attention_mask)
            po_preds = nn.Sigmoid()(po_logits.reshape(passage_ids.size(0), passage_ids.size(1), self.event_num, self.classes_num, 2))

            subject_loss = self.loss_fct(sub_preds, subject_labels)
            subject_loss = subject_loss.mean(2)
            subject_loss = torch.sum(subject_loss * attention_mask.float()) / torch.sum(attention_mask.float())
            
            po_loss = self.bi_match_loss(po_preds, object_labels, attention_mask)

            loss = subject_loss + po_loss

            return loss

        else:
            # (batch_size, seq_len, 2)
            subject_preds = nn.Sigmoid()(self.subject_dense(trigger_represent))
            answer_list = list()
            subject_ids = []
            for qid, sub_pred, trigger_repre in zip(q_ids.cpu().numpy(),
                                     subject_preds.cpu().numpy(), trigger_represent.cpu().numpy()):
                context = eval_file[qid].bert_tokens
                # start:(true_number, )
                start = np.where(sub_pred[:, 0] > config.eval_config["sub_threshold_start"])[0]
                end = np.where(sub_pred[:, 1] > config.eval_config["sub_threshold_end"])[0]
                if start.size == 0 or end.size == 0:
                    subject_ids.append((-1, -1))
                    answer_list.append(np.zeros((trigger_repre.shape[1] * 2)))
                    continue
                subject = []
                s_starts = []
                s_ends = []
                for i in start:
                    j = end[end >= i]
                    if i == 0 or i > len(context) - 2:
                        continue

                    if len(j) > 0:
                        j = j[0]
                        if j > len(context) - 2:
                            continue
                        subject.append((i, j))
                        s_starts.append(trigger_repre[i, :])
                        s_ends.append(trigger_repre[j, :])
                if not s_starts or not s_ends:
                    subject_ids.append((-1, -1))
                    answer_list.append(np.zeros((trigger_repre.shape[1] * 2)))
                    continue
                subject_ids.append(subject)
                s_starts = np.average(np.array(s_starts), axis=0)
                s_ends = np.average(np.array(s_ends), axis=0)
                answer_list.append(np.concatenate((s_starts, s_ends), axis=0))

            sub_tensor = torch.tensor(answer_list, dtype=torch.float32).unsqueeze(1).cuda()
            att_weight = self.additive_attention(argument_represent, sub_tensor, attention_mask).unsqueeze(2)
            context_encoder = argument_represent * att_weight
            context_encoder = torch.cat([context_encoder, argument_represent], dim=2)
            context_encoder = self.context_dense(context_encoder)
            po_logits = self.event_decoder(context_encoder, attention_mask)
            po_preds = nn.Sigmoid()(po_logits.reshape(passage_ids.size(0), passage_ids.size(1), self.event_num, self.classes_num, 2))

            return q_ids, po_preds
