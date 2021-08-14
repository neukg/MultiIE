import os
import json
import random
import numpy as np 
import torch
import config

if 'nezha' in config.bert_config["bert_path"]:
    print('using NEZHA...')
    # NEZHA
    from nezha.modeling_nezha import BertConfig, BertModel
    from nezha.main import torch_init_model
elif 'roformer' in config.bert_config["bert_path"]:
    print('using roformer...')
    from transformers.models.bert.modeling_bert import BertConfig
    from roformer import RoFormerModel, RoFormerConfig
else:
    from transformers import BertModel, AutoModel, AutoTokenizer

from data_loader import Reader, Feature
from train import Trainer
from utils.data_util import NewBertTokenizer
from model.casrel import ERENet

def build_dataset(fold:int, reader:Reader, train_max_len:int, dev_max_len:int, tokenizer, train_batch_size:int, dev_batch_size:int, rel2id:dict):

    train_data = []
    dev_data = []
    data_path = os.path.join(config.data_out_dir, config.common["exp_name"])
    
    for i in range(10):
        if i == fold:
            continue
        with open(os.path.join(data_path, 'data_{}.json'.format(i)), 'r') as f:
            train_data.extend(json.load(f))

    with open(os.path.join(data_path, 'data_{}.json'.format(fold)), 'r') as f:
        dev_data.extend(json.load(f)) 


    train_examples = reader.read_examples(train_data, 'train', config.train_config["max_len"])
    dev_examples = reader.read_examples(dev_data, 'dev', config.eval_config["max_len"])

    train_convert_examples_to_features = Feature(max_len=train_max_len, tokenizer=tokenizer)
    dev_convert_examples_to_features = Feature(max_len=dev_max_len, tokenizer=tokenizer)

    train_features = train_convert_examples_to_features(train_examples, rel2id, 'train')
    dev_features = dev_convert_examples_to_features(dev_examples, rel2id, 'dev')

    train_dataloader = train_features.get_dataloader(train_batch_size, num_workers=2, shuffle=True)
    dev_dataloader = dev_features.get_dataloader(dev_batch_size, num_workers=2, shuffle=True)

    data_loaders = train_dataloader, dev_dataloader
    eval_examples = train_examples, dev_examples
    return data_loaders, eval_examples

def main():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    random.seed(config.common["seed"])
    np.random.seed(config.common["seed"])
    torch.manual_seed(config.common["seed"])

    with open(os.path.join(config.data_out_dir, config.common["exp_name"], 'tag2id.json'), 'r') as f:
        rel2id = json.load(f)


    tokenizer = NewBertTokenizer.from_pretrained(config.bert_config["bert_path"], add_special_tokens = False, do_lower_case = True)
    tokenize = tokenizer.tokenize
    tok2char_span = lambda text: tokenizer.get_offset_mappings(text)

    reader = Reader(tokenize, tok2char_span, rel2id)

    data_loaders, eval_examples = build_dataset(fold=9, 
                                                reader=reader, 
                                                train_max_len=config.train_config["max_len"],
                                                dev_max_len=config.eval_config["max_len"],
                                                tokenizer=tokenizer,
                                                train_batch_size=config.train_config["batch_size"],
                                                dev_batch_size=config.eval_config["batch_size"],
                                                rel2id=rel2id
    )
    '''
    num = -1
    for i in eval_examples[0]:
        num += 1
        if num == 20:
            break
        
        print(i)

    num = -1
    for i in eval_examples[1]:
        num += 1
        if num == 6:
            break
        print()
        print(i)

    
    '''
    if 'nezha' in config.bert_config["bert_path"]:
        bert_config = BertConfig.from_json_file(config.bert_config["bert_path"]+'/config.json')
        bert_model = BertModel(bert_config)
        torch_init_model(bert_model, config.bert_config["bert_path"]+'/pytorch_model.bin')
    elif 'electra' in config.bert_config["bert_path"]:
        print('using ernie or electra')
        bert_model = AutoModel.from_pretrained(config.bert_config["bert_path"])
    elif 'roformer' in config.bert_config["bert_path"]:
        print('using roformer...')
        bert_model = RoFormerModel.from_pretrained(config.bert_config["bert_path"])
    else:
        bert_model = BertModel.from_pretrained(config.bert_config["bert_path"])

    encoder = ERENet(bert_model, classes_num=len(rel2id))
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
        trainer.eval_data_set("dev")
        badcase, type_error, bound_error = trainer.badcase("dev")
        bad_path = os.path.join('./', 'badcase.json')
        with open(bad_path, 'w') as f:
            for line in badcase:
                f.write(json.dumps(line, ensure_ascii = False ) + '\n')
        print("类型错误的共有： {}， 边界错误的共有： {}".format(type_error, bound_error))
    
if __name__ == '__main__':

    main()