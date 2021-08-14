# -*- encoding: utf-8 -*-

import re
import codecs
import json
import logging
import random
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.data_util import search, cut_sentence, argument_in_sent


class Example(object):
    def __init__(self,
                 p_id=None,
                 id = None,
                 context=None,
                 bert_tokens=None,
                 sub_pos=None,
                 sub_entity_list=None,
                 relative_pos_start=None,
                 relative_pos_end=None,
                 po_list=None,
                 gold_answer=None,
                 token_ids=None,
                 spoes=None,
                 text_char_span=None
    ):
        # read_example
        self.p_id = p_id
        self.id = id
        self.context = context
        self.sub_entity_list = sub_entity_list
        self.gold_answer = gold_answer
        self.spoes = spoes
        self.text_char_span = text_char_span

        self.bert_tokens = bert_tokens
        self.sub_pos = sub_pos
        self.relative_pos_start = relative_pos_start
        self.relative_pos_end = relative_pos_end
        self.po_list = po_list
        self.token_ids = token_ids

    def __repr__(self):
        return "pid: {}, text: {}, subject_list: {}, gold_answer: {}, spoes: {} \n text_char_span: {}".format(self.p_id, 
                                                                                             self.context, 
                                                                                             self.sub_entity_list, 
                                                                                             self.gold_answer, 
                                                                                             self.spoes,
                                                                                             self.text_char_span)

class Reader(object):
    def __init__(self, tokenize, tok2char_span, rel2id):
        self.tokenize = tokenize
        self.tok2char_span = tok2char_span
        self.rel2id = rel2id

    def read_examples(self, data, data_type, max_len, sliding_window):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(data, data_type, max_len, sliding_window)

    def _read(self, data, data_type, max_len, sliding_window):

        examples = []
        p_id = 0
        for data_json in tqdm(data):
            p_id += 1
            id = data_json["id"]
            original_text = data_json['text']
            title = data_json["title"]

            split_char_spans = []
            split_contexts = []
            split_bert_tokens = []
            # title
            split_title, split_title_start = cut_sentence(title, max_len - 2, sliding_window)
            for t, s_t in zip(split_title, split_title_start):
                title_char_spans = self.tok2char_span(t)
                split_char_spans.append(title_char_spans)
            split_contexts.extend(split_title)
            # text
            split_texts, split_starts = cut_sentence(original_text, max_len - 2, sliding_window)
            for split_text, split_start in zip(split_texts, split_starts):
                char_spans = self.tok2char_span(split_text)
                split_char_spans.append(char_spans)
            split_contexts.extend(split_texts)
            # [-1, -1]分别对应CLS , SEP
            for i, char_span in enumerate(split_char_spans):
                split_char_spans[i] = [[-1, -1]] + char_span + [[-1, -1]]
            for s_text in split_contexts:
                bert_tokens = ['[CLS]'] + self.tokenize(s_text) + ['[SEP]']
                split_bert_tokens.append(bert_tokens)
            
            if data_type == 'test':
                for i, bert_tokens in enumerate(split_bert_tokens):
                    examples.append(
                        Example(
                            p_id=p_id,
                            id = id,
                            context=split_contexts[i],
                            text_char_span = split_char_spans[i],
                            bert_tokens=bert_tokens
                        )
                    )
                continue

            # 给每个分句重新标注事件
            split_sents_event = []
            for sent in split_contexts:
                new_event_list = []
                if 'event_list' in data_json:
                    for event in data_json['event_list']:
                        tri_start, argus = argument_in_sent(sent, event["arguments"], event["trigger"])
                        if tri_start < 0 or len(argus) == 0:
                            continue
                        else:
                            new_event = {
                                "arguments": argus,
                                "event_type": event["event_type"],
                                "trigger": event["trigger"],
                                "trigger_start_index": tri_start
                            }
                            new_event_list.append(new_event)
                split_sents_event.append(new_event_list)
            
            # 给每个分句产生对应的三元组，及其对应的位置
            split_sub_ent_list, split_spo_list, split_so_ind_list = [], [], []
            for sent_event_list in split_sents_event:
                sub_ent_list, spo_list, so_ind_list = [], [], []
                for sent_event in sent_event_list:
                    subject_name = sent_event["trigger"]
                    sub_ent_list.append(subject_name)
                    spo, so_ind = [], []
                    for argument in sent_event["arguments"]:
                        object_name = argument["argument"]
                        relation = sent_event["event_type"] + "_" + argument["role"]
                        spo.append((subject_name, relation, object_name))
                        so_ind.append((sent_event["trigger_start_index"], argument["argument_start_index"]))
                    spo_list.append(spo)
                    so_ind_list.append(so_ind)
                split_sub_ent_list.append(sub_ent_list)
                split_spo_list.append(spo_list)
                split_so_ind_list.append(so_ind_list)

            spoes_list = []
            for spo_lists, so_ind_lists, text_char_span in zip(split_spo_list, split_so_ind_list, split_char_spans):
                spoes = []
                for event_spo, event_so_ind in zip(spo_lists, so_ind_lists):
                    event_spoes = []
                    for gold_answer, so_ind_list in zip(event_spo, event_so_ind):
                        s, p, o, s_start_ind, o_start_ind = (*gold_answer, *so_ind_list)

                        # 去掉首尾不可见字符
                        s_start_ind = s_start_ind + re.search(r'[^\s]', s).span()[0]
                        o_start_ind = o_start_ind + re.search(r'[^\s]', o).span()[0]
                        s = s.strip()
                        o = o.strip()

                        s_end_ind = s_start_ind + len(s)
                        o_end_ind = o_start_ind + len(o)

                        s = self.tokenize(s)
                        p = self.rel2id[p]
                        o = self.tokenize(o)

                        s_idx = search(text_char_span, s_start_ind, s_end_ind)
                        o_idx = search(text_char_span, o_start_ind, o_end_ind)


                        if s_idx != -1 and o_idx != -1:
                            s = (s_idx, s_idx + len(s) - 1)
                            o = (o_idx, o_idx + len(o) - 1, p)
                            event_spoes.append((s, o))
                    spoes.append(event_spoes)
                spoes_list.append(spoes)

            for i, spoes in enumerate(spoes_list):
                examples.append(
                    Example(
                        p_id=p_id,
                        id = id,
                        context=split_contexts[i],
                        sub_entity_list=split_sub_ent_list[i],
                        gold_answer=split_spo_list[i],
                        spoes=spoes,
                        text_char_span = split_char_spans[i],
                        bert_tokens=split_bert_tokens[i]
                    )
                )

        print("{} total size is  {} ".format(data_type, len(examples)))

        return examples

class Feature(object):
    def __init__(self, max_len, tokenizer):

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __call__(self, examples:Example, rel2id, event_num, data_type):

        return self.convert_examples_to_bert_features(examples, rel2id, event_num, data_type)

    def convert_examples_to_bert_features(self, examples:Example, rel2id, event_num, data_type):

        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, data_type=data_type, rel2id=rel2id, event_num=event_num,
                          tokenizer=self.tokenizer, max_len=self.max_len)

class SPODataset(Dataset):
    def __init__(self, data, data_type, rel2id, event_num, tokenizer=None, max_len=128):
        super(SPODataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.q_ids = [f[0] for f in data]
        self.features = [f[1] for f in data]
        self.is_train = True if data_type == 'train' else False
        self.rel2id = rel2id
        self.predict_num = len(rel2id)
        self.event_num = event_num

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.features[index]

    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids, batch_segment_ids, batch_attention_mask = [], [], []
            batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
            batch_event_num = []
            for example in examples:
                # todo maxlen 
                codecs = self.tokenizer(example.context,
                                        add_special_tokens = True,
                                        max_length = self.max_len, 
                                        truncation = True,
                                        padding = 'max_length')
                token_ids, segment_ids, attention_masks = codecs["input_ids"], codecs["token_type_ids"], codecs["attention_mask"]

                example.token_ids = token_ids
                # pp = self.tokenizer.tokenize(example.context)
                # ppp = self.tokenizer.decode(token_ids)

                if self.is_train:
                    spoes = example.spoes
                    
                    # subject标签
                    subject_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
                    # 对应的object标签
                    object_labels = np.zeros((len(token_ids), self.event_num, self.predict_num, 2), dtype=np.float32)
                    if spoes:
                        for index, event_spo in enumerate(spoes):
                            
                            for spo in event_spo:
                                s, po = spo
                                subject_labels[s[0], 0] = 1
                                subject_labels[s[1], 1] = 1
                                
                                object_labels[po[0], index, po[2], 0] = 1
                                object_labels[po[1], index, po[2], 1] = 1
                    batch_token_ids.append(token_ids)
                    batch_attention_mask.append(attention_masks)
                    batch_segment_ids.append(segment_ids)
                    batch_subject_labels.append(subject_labels)
                    batch_object_labels.append(object_labels)

                else:
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    batch_attention_mask.append(attention_masks)

            if not self.is_train:
                batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
                batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long)
                batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
                # pids:(batch_size,); batch_token_ids:(batch_size, seq_len); batch_segment_ids:(batch_size, seq_len)
                return p_ids, batch_token_ids, batch_segment_ids, batch_attention_mask
            else:
                batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
                batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long)
                batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
                # batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.long)
                batch_subject_labels = torch.tensor(batch_subject_labels, dtype=torch.float32)
                batch_object_labels = torch.tensor(batch_object_labels, dtype=torch.float32)
                batch_event_num = torch.tensor(batch_event_num, dtype=torch.float32)
                return batch_token_ids, batch_segment_ids, batch_attention_mask, batch_subject_labels, batch_object_labels, batch_event_num

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)