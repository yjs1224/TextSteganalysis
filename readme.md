# 文本隐写分析 (linguistic steganalysis)


# NOTICE: 之前版本可能存在一些严重的bugs,请自己检查使用的版本。 Serious Bugs in older version (Please Check your versions):
- #### 最好的模型可能会被较差的模型覆盖（已修复）
- #### 训练集上的metircs缺失（已修复）
- #### 以不正确的比率拆分train和val（固定）
- #### 添加了一些重要的注意事项（使用此repo之前请仔细阅读）
- #### The best model may be covered by worse model (Fixed)
- #### metircs on train set are missed (Fixed)
- #### split train and val in incorrect ratio (Fixed)
- #### Add Attention. (Carefully read it before using this repo)

### 感谢发现指出问题！
### Thanks for finding and reporting bugs!


## 测试环境 (environment for development)
python 3.7

transformers 4.12

## 包含（include）
### Non-Transformer-based
- [TS-CSW: text steganalysis and hidden capacity estimation based on convolutional sliding windows (TS-CSW)](https://link.springer.com/article/10.1007/s11042-020-08716-w)
- [TS-RNN: Text Steganalysis Based on Recurrent Neural Networks (TS-BiRNN)](https://ieeexplore.ieee.org/abstract/document/8727932)
- [Linguistic Steganalysis via Densely Connected LSTM with Feature Pyramid (BiLISTM-Dense)](https://dl.acm.org/doi/abs/10.1145/3369412.3395067)
- [A Fast and Efficient Text Steganalysis Method (TS-FCN)](https://ieeexplore.ieee.org/document/8653856)
- [A Hybrid R-BILSTM-C Neural Network Based Text Steganalysis(R-BiLSTM-C)](https://ieeexplore.ieee.org/abstract/document/8903243)
- [Linguistic Steganalysis With Graph Neural Networks (GNN) (waiting......) ](https://ieeexplore.ieee.org/document/9364681)

#### multi-stages
- [Real-Time Text Steganalysis Based on Multi-Stage Transfer Learning (MS-TL)](https://ieeexplore.ieee.org/abstract/document/9484749/)
------
### Transformer-based
- [SeSy: Linguistic Steganalysis Framework Integrating Semantic and Syntactic Features (Sesy)](https://ieeexplore.ieee.org/abstract/document/9591452)
- [High-Performance Linguistic Steganalysis, Capacity Estimation and Steganographic Positioning (BERT-LSTM-ATT)](https://link.springer.com/chapter/10.1007%2F978-3-030-69449-4_7)

## 使用样例（How to use）
- 使用命令行：`python main.py --config_path your_config_file_path`
- 例如：`python main.py --config_path ./configs/test.json`
- 作为参考，./configs/test.json 里包含多种方法的超参数设置（存在冗余，根据自己需求删改），其中"use_plm"用来控制是否需要预训练语言模型，"model"表示使用哪种方法，根据需求修改"Training_with_Processor"或者"Training"参数，推荐前者即使用Processor进行预处理。默认情况下，我们只设置了**num_train_epochs=1, 在很多数据集下，是不能很好收敛的**。
- 例如，如果想使用[TS-FCN](https://link.springer.com/article/10.1007/s11042-020-08716-w)方法，一个最简单的方法就是将test.json里的"model"改为"fcn","use_plm"改为false, 然后设置合适的学习率、epoch、batchsize 等等(在Training_with_Processor中)
- Use Command: `python main.py --config_path your_config_file_path`
- For example: `python main.py --config_path ./configs/test.json`
- As a reference, ./configs/test.json contains the hyper-parameters settings of various methods (redundant, delete and modify up to your own setting). Among them, "use_plm" is used to control whether the pretrained language model (PLM) is used to embed words. "model" indicates which method to be use. Modify the "training_with_processor" or "training" parameters according to your needs. It is recommended that the former, namely using processor for pre-processing your data. By default, we only set **num_ Train_ Epochs = 1, where models can not converge well in many datasets **.
- For example, if you want to try the [TS-FCN](https://link.springer.com/article/10.1007/s11042-020-08716-w) method, a very easy way is to set "model" as "fcn", set "use_plm" as false, and then set the appropriate learning rate, epoch, batchsize, etc. (in Training_with_Processor), based on "test.json".

## 注意事项(Attention!)
- 之前的sesy config不是[论文](https://ieeexplore.ieee.org/abstract/document/9591452)中使用的config. 已修复(2023.03)。
- 如果使用此代码进行不同方法的对比时，请仔细确认数据集的拆分方式是否相同。
  - 可以在config.json 的Dataset中设置自己的“split_ratio”，例如设置为0.8, 则会产生0.2的test.csv”, 0.2的“val.csv”和0.6的“train.csv”
  - 或者也可以将“stego_txt”和“cover_txt”设置为None，设置“resplit"为False，并将数据集放在“csv_dir”所指示的文件路径中。放置的数据集应根据对比实验的需要，事先自行拆分为“train.csv”、“val.csv”和“test.csv”。
- Previous sesy config is not the configs used in paper [SESY](https://ieeexplore.ieee.org/abstract/document/9591452). We have fixed it.
- When using this repo for camparation, please carefully confirm whether the spliting of data set is the same. 
  - You can reset the "split_ratio" in Dataset config. For example, if set it to 0.8, you will get 0.2 "test.csv", 0.2 "val.csv", and 0.6 "train. csv".
  - Or, you can set stego_txt and cover_txt as None, set "resplit" as False, and put your dataset in path "csv_dir". The dataset should beforehand be split into "train.csv", "val.csv" and "test.csv" by yourself.


## Codes Reference
基础 [R-BiLSTM-C](https://ieeexplore.ieee.org/abstract/document/8903243) [BiLISTM-Dense](https://dl.acm.org/doi/abs/10.1145/3369412.3395067) [MS-TL](https://ieeexplore.ieee.org/abstract/document/9484749/) 模型的实现参考自[CAU-Tstega/Text-steganalysis](https://github.com/CAU-Tstega/Text-steganalysis)

implements of [R-BiLSTM-C](https://ieeexplore.ieee.org/abstract/document/8903243) [BiLISTM-Dense](https://dl.acm.org/doi/abs/10.1145/3369412.3395067) [MS-TL](https://ieeexplore.ieee.org/abstract/document/9484749/) refer to [CAU-Tstega/Text-steganalysis](https://github.com/CAU-Tstega/Text-steganalysis)

# 欢迎补充与讨论 (both supplement and discussion are grateful and helpful)
