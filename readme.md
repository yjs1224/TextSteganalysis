# 文本隐写分析 (linguistic steganalysis)

# NOTICE
## Serious Bugs in older version (Please Check your versions):
- #### The best model may be covered by worse model (Fixed)
- #### metircs on train set are missed (Fixed)
- #### split train and val in incorrect ratio (Fixed)

### Thanks for finding more bugs


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


## Codes Reference
基础 [R-BiLSTM-C](https://ieeexplore.ieee.org/abstract/document/8903243) [BiLISTM-Dense](https://dl.acm.org/doi/abs/10.1145/3369412.3395067) [MS-TL](https://ieeexplore.ieee.org/abstract/document/9484749/) 模型的实现参考自[CAU-Tstega/Text-steganalysis](https://github.com/CAU-Tstega/Text-steganalysis)

implements of [R-BiLSTM-C](https://ieeexplore.ieee.org/abstract/document/8903243) [BiLISTM-Dense](https://dl.acm.org/doi/abs/10.1145/3369412.3395067) [MS-TL](https://ieeexplore.ieee.org/abstract/document/9484749/) refer to [CAU-Tstega/Text-steganalysis](https://github.com/CAU-Tstega/Text-steganalysis)

# 欢迎补充与讨论 (both supplement and discussion are grateful and helpful)
