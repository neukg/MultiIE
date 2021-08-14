data_in_dir = '../duee_fin'
data_out_dir = '../data4bert'

train_mode = "train"

common = {
    "exp_name": "DuEE_fin",
    "device_num": 0,
    "model_output_path": "./model_dict/",
    "seed": 233,
    "device_id": 0,
    "event_num": 16,    # 设定的最大事件数
    "decoder_layer": 3  # decoder layer的层数
}

train_config = {
    "batch_size": 18,
    "epoch_num":  10,
    "learning_rate": 3e-5,
    "patience": 6,
    "warmup_proportion": 0.01,
    "max_grad_norm": 2,
    "max_len": 512,
    "sliding_window": 512
}

eval_config = {
    "batch_size": 1,
    "max_len": 512,
    "sliding_window": 400,
    "sub_threshold_start": 0.6,
    "sub_threshold_end": 0.5,
    "obj_threshold_start": 0.6,
    "obj_threshold_end": 0.5,
    "dev_epoch": 9
}

bert_config = {
    "data_home": "../data4bert",
    "bert_path": '../pretrained_models/FinBERT_L-12_H-768_A-12_pytorch', # bert-base-cased， chinese-bert-wwm-ext-hit, FinBERT_L-12_H-768_A-12_pytorch
    "hyper_parameters": {
        "lr": 5e-5,
    },
}