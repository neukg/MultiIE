## Get Started

##### Environment

* python==3.6.0

* torch==1.6.0
* transformers==3.0.0

##### Dataset&Data preparation

Please download the following datasets:

* [DuEE1.0](https://aistudio.baidu.com/aistudio/competition/detail/46/?isFromLUGE=TRUE)

You can move it to '../DuEE' folder that corresponds to config.data_in_dir.

For data preparation, run data_analysis.ipynb to preprocess data, and then the processed data is created into '../data4bert' that corresponds to config.data_out_dir. 

##### Config

The configuration is wrote in config.py. You can modify it for your convenience.

The meaning of key parameter is followed:

* train_mode: the mode of run. 'train' or 'test'.
* common.model_output_path: The saved path of model state_dict.
* bert_config.bert_path: The path of pretrained language model.

## Run

##### Train

To modify train_mode to 'train' in config.py, and run main.py.

##### Prediction

To modify train_model to 'test' in config.py, and run test.py.

##### Notice

We only provide the code for single model, the ensemble code can't be found in here.