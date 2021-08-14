from .modeling_nezha import BertConfig
from .modeling_nezha import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch


def torch_init_model(model, init_checkpoint, delete_module=False):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    state_dict_new = {}
    # delete module.
    if delete_module:
        for key in state_dict.keys():
            v = state_dict[key]
            state_dict_new[key.replace('module.', '')] = v
        state_dict = state_dict_new
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.bert = BertModel(config) #娉ㄦ剰淇敼
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self,token2id, token_mask):
        embed = self.bert(input_ids=token2id.long(),attention_mask=token_mask.long())
        return self.dropout(embed[0])

'''
bert_config_path="/data0/liuyaduo/pretrained_models/nezha-base-wwm/config.json"
nezha_path="/data0/liuyaduo/pretrained_models/nezha-base-wwm/pytorch_model.bin"

config = BertConfig.from_json_file(bert_config_path)
train_model = Model(config)
torch_init_model(train_model, nezha_path)

token2id=torch.tensor([
    [1,2,3],
    [4,5,6]
])
mask=torch.tensor([
    [1,1,1],
    [1,0,0]
])

embed=train_model(token2id, mask)
print(embed.shape) #torch.Size([2, 3, 1024])
'''