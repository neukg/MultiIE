# _*_ coding:utf-8 _*_
import logging
import os
import sys
import time

import numpy as np
import torch
import model.casrel as casrel
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import MultiStepLR

import config
from utils.train_util import FGM, get_multi_step_with_warmup_scheduler

logger = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self, encoder, data_loaders, examples, spo_conf, seed, device_id, output_dir):
        print('using ad')
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = 1

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf
        
        
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
        
        self.model = encoder
        self.model.to(self.device)

        '''
        logging.info('total gpu num is {}'.format(self.n_gpu))
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])
        else:
            self.model = nn.DataParallel(self.model.cuda())
        '''
        self.resume(output_dir) 
        
        self.adversarial_train = FGM(self.model)

        if len(data_loaders) == 2:
            train_dataloader, dev_dataloader = data_loaders
            train_eval, dev_eval = examples
            self.eval_file_choice = {
                "train": train_eval,
                "dev": dev_eval,
            }
            self.data_loader_choice = {
                "train": train_dataloader,
                "dev": dev_dataloader,
            }
            self.max_grad_norm = config.train_config["max_grad_norm"]
            num_training_steps = (int(len(train_eval) / config.train_config["batch_size"]) + 1) * config.train_config["epoch_num"]
            num_warmup_steps = int(config.train_config["warmup_proportion"] * num_training_steps)
            self.optimizer = AdamW(self.model.parameters(), lr=config.train_config["learning_rate"], correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
            #self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_warmup_steps)
            
            #self.scheduler = get_multi_step_with_warmup_scheduler(self.optimizer, [5, 10, 20], 0.5, -1)
        else:
            test_dataloader = data_loaders
            test_eval = examples
            self.eval_file_choice = {
                "test": test_eval,
            }
            self.data_loader_choice = {
                "test": test_dataloader,
            }

    def train(self, epoch_num, patience, model_output_path):

        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        step_gap = 20
        for epoch in range(epoch_num):

            global_loss = 0.0

            for step, batch in tqdm(enumerate(self.data_loader_choice[u"train"]), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):

                loss = self.forward(batch)

                global_loss += loss
                if step % step_gap == 0:
                    current_loss = global_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.data_loader_choice["train"]),
                                                                           epoch, current_loss))
                    global_loss = 0.0

            res_dev = self.eval_data_set("dev")

            if res_dev['f1'] >= best_f1:
                best_f1 = res_dev['f1']
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = self.model.module if hasattr(self.model,
                                                             'module') else self.model  # Only save the model it-self
                output_model_file = model_output_path + "/pytorch_model.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))
                patience_stop = 0
            else:
                patience_stop += 1
       

    def resume(self, output_dir):
        resume_model_file = output_dir + "/pytorch_model.bin"
        if os.path.exists(resume_model_file):
            logging.info("=> loading checkpoint '{}'".format(resume_model_file))
            checkpoint = torch.load(resume_model_file, map_location='cpu')
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
        else:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

    def forward(self, batch, chosen=u'train', eval=False, answer_dict=None):

        batch = tuple(t.to(self.device) for t in batch)
        if not eval:
            input_ids, segment_ids, attention_mask, subject_labels, object_labels = batch

            loss = self.model(passage_ids=input_ids,
                              segment_ids=segment_ids,
                              attention_mask=attention_mask,
                              subject_labels=subject_labels,
                              object_labels=object_labels)

            
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            loss.backward()
            self.adversarial_train.attack()
            
            loss_adv =self.model(passage_ids=input_ids, 
                                 segment_ids=segment_ids,
                                 attention_mask=attention_mask,
                                 subject_labels=subject_labels, 
                                 object_labels=object_labels)
            
            if self.n_gpu > 1:
                loss_adv = loss_adv.mean()
            
            loss_adv.backward()
            self.adversarial_train.restore()

            loss = loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, input_ids, segment_ids, attention_mask = batch
            eval_file = self.eval_file_choice[chosen]
            qids, subject_pred, po_pred = self.model(q_ids=p_ids,
                                                     passage_ids=input_ids,
                                                     segment_ids=segment_ids,
                                                     attention_mask=attention_mask,
                                                     eval_file=eval_file, is_eval=eval
            )
            # qids:(all_subject_length); subject_pred:(all_subject_ids, 2); po_pred:(all_subject_length, seq_len, class_num, 2);
            self.convert_spo_contour(qids,
                                     po_pred,
                                     eval_file,
                                     answer_dict, 
                                     use_bert=True
            )
            return answer_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {example.id: [[], []] for example in eval_file}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        if chosen == "dev":
            res = self.evaluate(eval_file, answer_dict, chosen)
        if chosen == "test":
            res = self.convert_answerdict_to_test(eval_file, answer_dict)

        self.model.train()
        return res

    def badcase(self, chosen="dev"):
        
        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        if chosen == "dev":
            bad_case = []
            all_type_error = 0
            all_bound_error = 0
            for key, value in answer_dict.items():
                pred_score = 1e-12
                all_pred_num = 1e-12
                all_gold_num = 1e-12

                type_error = 0
                bound_error = 0

                triple_preds, triple_golds = value

                triple_golds = set(triple_golds)
                triple_preds = set(triple_preds)

                all_pred_num += len(triple_preds)
                all_gold_num += len(triple_golds)
                for gold_triple in triple_golds:
                    max_score = 0
                    for pred_triple in triple_preds:
                        if pred_triple[0] == gold_triple[1]:
                            score = Trainer.evaluate_word(pred_triple[1], gold_triple[2])
                            if score > max_score:
                                max_score = score
                    if max_score == 0:
                        type_error += 1
                    elif max_score < 0.999:
                        bound_error += 1
                    pred_score += max_score

                all_type_error += type_error
                all_bound_error += bound_error

                precision = pred_score / all_pred_num
                recall = pred_score / all_gold_num
                f1 = 2 * precision * recall / (precision + recall)
                if f1 < 0.9999:
                    bad_case.append(
                        {
                            'pred': list(triple_preds),
                            'gold': list(triple_golds),
                            'bound_error': bound_error,
                            'type_error': type_error
                        }
                    )

        self.model.train()
        return bad_case, all_type_error, all_bound_error

    def convert_answerdict_to_test(self, eval_file, answer_dict):

        test_dict_list = []
        for key, value in answer_dict.items():
            event_dict = {}
            id = key
            triple_preds, _ = value

            for pred_triple in triple_preds:
                event_type, role = pred_triple[0].split('_')
                argument = (role, pred_triple[1])
                if event_type not in event_dict:
                    event_dict[event_type] = set([argument])
                else:
                    event_dict[event_type].add(argument)

            event_list = []
            for event_type, arguments in event_dict.items():
                event_list.append(
                    {
                        "event_type": event_type,
                        "arguments": [{"role": argument[0], "argument": argument[1]} for argument in arguments]
                    }
                )
            test_dict_list.append(
                {
                    "id": id,
                    "event_list": event_list
                }
            )
        return test_dict_list

    @staticmethod
    def evaluate_word(pred, gold):
        pred_num = len(pred) + 1e-12
        gold_num = len(gold) + 1e-12
        common_num = 1e-12
        for char in pred:
            if char in gold:
                common_num += 1
        precision = common_num / pred_num
        if common_num < gold_num:
            recall = common_num / gold_num
        else:
            recall = 1
        f1 = 2 * precision * recall / (precision + recall)
        return f1
        
    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):
        
        pred_score = 1e-12
        all_pred_num = 1e-12
        all_gold_num = 1e-12

        for key, value in answer_dict.items():
            triple_preds, triple_golds = value

            triple_golds = set(triple_golds)
            triple_preds = set(triple_preds)

            all_pred_num += len(triple_preds)
            all_gold_num += len(triple_golds)
            for gold_triple in triple_golds:
                max_score = 0
                for pred_triple in triple_preds:
                    if pred_triple[0] == gold_triple[1]:
                        score = Trainer.evaluate_word(pred_triple[1], gold_triple[2])
                        if score > max_score:
                            max_score = score
                pred_score += max_score

        precision = pred_score / all_pred_num
        recall = pred_score / all_gold_num
        f1 = 2 * precision * recall / (precision + recall)

        print('============================================')
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1 * 100, precision * 100,
                                                                recall * 100))
        return {'f1': f1, "recall": recall, "precision": precision}

    def convert_spo_contour(self, qids, po_preds, eval_file, answer_dict, use_bert=False):

        for qid, po_pred in zip(qids.data.cpu().numpy(),
                                         po_preds.data.cpu().numpy()):
            if qid == -1:
                continue
            tokens = eval_file[qid.item()].bert_tokens
            token_ids = eval_file[qid.item()].token_ids
            id = eval_file[qid.item()].id
            gold_answer = eval_file[qid.item()].gold_answer
            # 用于抽取原文中的实体
            text = eval_file[qid.item()].context
            text_char_span = eval_file[qid.item()].text_char_span
            # (seq_len, class_num)
            start = np.where(po_pred[:, :, 0] > config.eval_config["obj_threshold_start"])
            end = np.where(po_pred[:, :, 1] > config.eval_config["obj_threshold_end"])

            spoes = []
            for _start, predicate1 in zip(*start):
                if _start > len(tokens) - 2 or _start == 0:
                    continue
                for _end, predicate2 in zip(*end):
                    if _start <= _end <= len(tokens) - 2 and predicate1 == predicate2:
                        spoes.append((predicate1, (_start, _end)))
                        break
            po_predict = []
            for p, o in spoes:
                try:
                    po_predict.append((self.id2rel[p],
                                    text[text_char_span[o[0]][0] : text_char_span[o[1]][-1]])
                    )
                except:
                    raise ValueError(text_char_span, o, text)

            if id not in answer_dict:
                print('erro in answer_dict ')
            else:
                answer_dict[id][0].extend(po_predict)
                if gold_answer:
                    if not answer_dict[id][1]:
                        answer_dict[id][1].extend(gold_answer)