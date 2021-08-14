import os
import json
import random
import config
import numpy as np

import torch
from transformers import BertModel

from utils.data_util import NewBertTokenizer
from model.casrel import ERENet
from train import Trainer
from data_loader import Reader, Feature


def build_dataset(fold:int, reader:Reader, test_max_len:int, tokenizer:NewBertTokenizer, test_batch_size:int, rel2id:dict):

 
    data_path = os.path.join(config.data_in_dir, 'duee_fin_test2.json')

    with open(data_path, 'r') as f:
        test_data = [json.loads(line) for line in f.readlines()]

    test_examples = reader.read_examples(test_data, 'test', config.eval_config["max_len"], config.eval_config["sliding_window"])

    convert_examples_to_features = Feature(max_len=test_max_len, tokenizer=tokenizer)

    test_features = convert_examples_to_features(test_examples, rel2id, config.common['event_num'], 'test')

    test_dataloader = test_features.get_dataloader(test_batch_size, num_workers=2, shuffle=False)

    data_loaders = test_dataloader
    eval_examples = test_examples
    return data_loaders, eval_examples

def main():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    random.seed(config.common["seed"])
    np.random.seed(config.common["seed"])
    torch.manual_seed(config.common["seed"])

    
    with open(os.path.join(config.data_out_dir, config.common["exp_name"], 'tag2id.json'), 'r') as f:
        rel2id = json.load(f)

    tokenizer = NewBertTokenizer.from_pretrained(config.bert_config["bert_path"], add_special_tokens = False, do_lower_case = True)
    tokenize = tokenizer.tokenize
    tok2char_span = lambda text: tokenizer.get_offset_mappings(text)

    reader = Reader(tokenize, tok2char_span, rel2id)

    data_loaders, eval_examples = build_dataset(fold=0, 
                                                reader=reader, 
                                                test_max_len=config.eval_config["max_len"],
                                                tokenizer=tokenizer,
                                                test_batch_size=config.eval_config["batch_size"],
                                                rel2id=rel2id
    )
    bert_model = BertModel.from_pretrained(config.bert_config["bert_path"])
    encoder = ERENet(bert_model, classes_num=len(rel2id), event_num=config.common["event_num"])

    trainer = Trainer(encoder=encoder,
                      data_loaders = data_loaders, 
                      examples = eval_examples, 
                      spo_conf=rel2id, 
                      seed=config.common["seed"], 
                      device_id=config.common["device_num"], 
                      output_dir=config.common["model_output_path"]
    )
    res = trainer.eval_data_set(chosen='test')

    answer_path = os.path.join('./test2', 'duee_fin_6_400.json')
    with open(answer_path, 'w') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii = False ) + '\n')
    print("---------写入完毕---------")

if __name__ == '__main__':

    main()