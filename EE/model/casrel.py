
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn.parameter import Parameter
from transformers import BertModel

import config


class ERENet(nn.Module):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, encoder, classes_num):
        super().__init__()
        self.classes_num = classes_num
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        layer_num = encoder.config.num_hidden_layers

        self.trigger_context_lstm = nn.LSTM(hidden_size, 
                                            hidden_size // 2, 
                                            num_layers = 1, 
                                            bidirectional = True, 
                                            batch_first = True)
        self.trigger_lstm_dropout = nn.Dropout(0.1)
        self.argument_context_lstm = nn.LSTM(hidden_size, 
                                             hidden_size // 2, 
                                             num_layers = 1, 
                                             bidirectional = True, 
                                             batch_first = True)
        self.argument_lstm_dropout = nn.Dropout(0.1)
        # pointer net work
        self.po_dense = nn.Linear(hidden_size * 2, self.classes_num * 2)
        self.subject_dense = nn.Linear(hidden_size, 2)

        # additive attention
        self.w1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w2 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.tanh = nn.Tanh()
        self.attn1 = nn.Linear(hidden_size // 2, 1)
        self.w3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # BERT动态融合
        # trigger
        #self.trigger_weight = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layer_num)])
        # argument 
        #self.argument_weight = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layer_num)])

        # 使用幂次惩罚
        # self.loss_fct = nn.BCELoss(reduction='none')
        self.loss_fct = nn.BCELoss(reduction='none')
    
    def dynamic_fusion(self, hidden_states, weights):
        logits = []
        out_hidden_states = []
        for i, weight in enumerate(weights):
            logits.append(weight(hidden_states[i]))
            out_hidden_states.append(hidden_states[i].unsqueeze(2))
        logits = torch.cat(logits, dim=2).cuda()
        # bs, seq_len, 1, 12
        out_weight = F.softmax(logits, dim=2).unsqueeze(2)
        # bs, sl, 12, hs
        hidden_states = torch.cat(out_hidden_states, dim=2).cuda()
        out_hidden = torch.matmul(out_weight, hidden_states).squeeze(2)
        # bs, sl, hs
        return out_hidden

    def additive_attention(self, h, g, mask=None):
        x = self.attn1(self.tanh(self.w1(h) + self.w2(g)))
        if mask != None:
            x = x * mask.unsqueeze(2)
        x = x.squeeze(-1)
        weight = F.softmax(x, 1)
        return weight

    def forward(self, q_ids=None, passage_ids=None, segment_ids=None, attention_mask=None, subject_labels=None, object_labels=None, 
                eval_file=None,
                is_eval=False):

        # bert_encoder(batch_size, seq_len, hidden_size)
        bert_encoder = self.encoder(input_ids=passage_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        # bert_encoder2, pooler_out2, hidden_states2 = self.encoder2(passage_ids, attention_mask, segment_ids, output_hidden_states=True)
        # 1: trigger: last two  argument: last one
        trigger_represent, _ = self.trigger_context_lstm(bert_encoder)
        trigger_represent = self.trigger_lstm_dropout(trigger_represent)
        argument_represent, _ = self.argument_context_lstm(bert_encoder)
        argument_represent = self.argument_lstm_dropout(argument_represent)
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

            # batch, hidden_size * 2
            subject = torch.cat([start_sub, end_sub], 2)

            att_weight = self.additive_attention(argument_represent, subject, attention_mask).unsqueeze(2)
            context_encoder = argument_represent * att_weight
            
            context_encoder = torch.cat([context_encoder, argument_represent], dim=2)
            #context_encoder = self.LayerNorm(bert_encoder, subject)

            sub_preds = nn.Sigmoid()(self.subject_dense(trigger_represent))

            po_preds = nn.Sigmoid()(self.po_dense(context_encoder).reshape(passage_ids.size(0), -1, self.classes_num, 2))

            # 使用幂次惩罚
            # sub_preds = nn.Sigmoid()(sub_preds) ** 2
            # po_preds = nn.Sigmoid()(po_preds) ** 2

            subject_loss = self.loss_fct(sub_preds, subject_labels)
            # subject_loss = F.binary_cross_entropy(F.sigmoid(sub_preds) ** 2, subject_labels, reduction='none')
            subject_loss = subject_loss.mean(2)
            subject_loss = torch.sum(subject_loss * attention_mask.float()) / torch.sum(attention_mask.float())

            po_loss = self.loss_fct(po_preds, object_labels)
            # po_loss = F.binary_cross_entropy(F.sigmoid(po_preds) ** 4, object_labels, reduction='none')
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * attention_mask.float()) / torch.sum(attention_mask.float())

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
            po_pred = self.po_dense(context_encoder).reshape(sub_tensor.size(0), -1, self.classes_num, 2)
            po_tensor = nn.Sigmoid()(po_pred)

            return q_ids, subject_ids, po_tensor
