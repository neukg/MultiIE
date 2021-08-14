import os
import json
import random
import numpy as np 
import torch

from transformers import BertModel

import config
from data_loader import Reader, Feature
from train import Trainer
from utils.data_util import NewBertTokenizer
from model.casrel import ERENet

def build_dataset(fold:int, reader:Reader, train_max_len:int, dev_max_len:int, tokenizer, train_batch_size:int, dev_batch_size:int, rel2id:dict):

    train_data = []
    dev_data = []
    data_path = os.path.join(config.data_out_dir, config.common["exp_name"])
    
    for i in range(7):
        if i == fold:
            continue
        with open(os.path.join(data_path, 'data_{}.json'.format(i)), 'r') as f:
            train_data.extend(json.load(f))

    with open(os.path.join(data_path, 'data_{}.json'.format(fold)), 'r') as f:
        dev_data.extend(json.load(f)) 


    train_examples = reader.read_examples(train_data, 'train', config.train_config["max_len"], config.train_config["sliding_window"])
    dev_examples = reader.read_examples(dev_data, 'dev', config.eval_config["max_len"], config.eval_config["sliding_window"])

    
    train_convert_examples_to_features = Feature(max_len=train_max_len, tokenizer=tokenizer)
    dev_convert_examples_to_features = Feature(max_len=dev_max_len, tokenizer=tokenizer)

    train_features = train_convert_examples_to_features(train_examples, rel2id, config.common['event_num'], 'train')
    dev_features = dev_convert_examples_to_features(dev_examples, rel2id, config.common['event_num'], 'dev')

    train_dataloader = train_features.get_dataloader(train_batch_size, num_workers=6, shuffle=True)
    dev_dataloader = dev_features.get_dataloader(dev_batch_size, num_workers=6, shuffle=False)

    data_loaders = train_dataloader, dev_dataloader
    
    eval_examples = train_examples, dev_examples
    
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
                                                train_max_len=config.train_config["max_len"],
                                                dev_max_len=config.eval_config["max_len"],
                                                tokenizer=tokenizer,
                                                train_batch_size=config.train_config["batch_size"],
                                                dev_batch_size=config.eval_config["batch_size"],
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

    if config.train_mode == "train":
        trainer.train(config.train_config["epoch_num"],
                      config.train_config["patience"],
                      config.common["model_output_path"]
        )
    if config.train_mode == "dev":
        res = trainer.eval_data_set("dev")
        answer_path = os.path.join('.', 'new_dev.json')
        with open(answer_path, 'w') as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii = False ) + '\n')
        print("---------写入完毕---------")
   

if __name__ == '__main__':

    main()