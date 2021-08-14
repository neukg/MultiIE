data_in_dir = '../DuEE'
data_out_dir = '../data4bert'

train_mode = "train"

common = {
    "exp_name": "DuEE",
    "device_num": 0,
    "model_output_path": "./model_dict/",
    "seed": 2021,
    "device_id": 0
}

train_config = {
    "batch_size": 10,
    "epoch_num":  40,
    "learning_rate": 5e-5,
    "patience": 6,
    "warmup_proportion": 0.025,
    "max_grad_norm": 1.0,
    "max_len": 300
}

eval_config = {
    "batch_size": 4,
    "max_len": 300,
    "sub_threshold_start": 0.6,
    "sub_threshold_end": 0.5,
    "obj_threshold_start": 0.6,
    "obj_threshold_end": 0.5,
    "dev_epoch": 9
}

bert_config = {
    "data_home": "../data4bert",
    "bert_path": '/data2/liuyaduo/pretrained_models/chinese-electra-base', # chinese-bert-wwm-ext-hit, nezha-base-www, 
    "hyper_parameters": {
        "lr": 5e-5,
    },
}