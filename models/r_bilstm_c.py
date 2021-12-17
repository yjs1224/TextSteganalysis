import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

class TC_base(nn.Module):
    def __init__(self,in_features, class_num, num_layers, hidden_size,Ci, kernel_num, kernel_sizes,LSTM_dropout,CNN_dropout):
        super(TC_base, self).__init__()
        self.in_features =in_features
        self.class_num = class_num
        D = in_features
        C = class_num
        N = num_layers
        H = hidden_size
        Ci = Ci
        Co = kernel_num
        Ks = kernel_sizes
        self.lstm = nn.LSTM(D, H, num_layers=N, \
                            bidirectional=True,
                            batch_first=True,
                            dropout=LSTM_dropout)

        self.conv1_D = nn.Conv2d(Ci, Co, (1, 2 * H))

        self.convK_1 = nn.ModuleList(
            [nn.Conv2d(Co, Co, (K, 1)) for K in Ks])

        self.conv3 = nn.Conv2d(Co, Co, (3, 1))

        self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1, 0))

        self.CNN_dropout = nn.Dropout(CNN_dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, features):
        out, _ = self.lstm(features)  # [batch_size, sen_len, H*2]
        x = out.unsqueeze(1)
        x = self.conv1_D(x)

        x = [F.relu(conv(x)) for conv in self.convK_1]
        x3 = [F.relu(self.conv3(i)) for i in x]
        x4 = [F.relu(self.conv4(i)) for i in x3]
        inception = []
        for i in range(len(x4)):
            res = torch.add(x3[i], x4[i])
            inception.append(res)

        x = [i.squeeze(3) for i in inception]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.CNN_dropout(x)
        logits = self.fc1(x)
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )



class TC(nn.Module):
    def __init__(self, embed_dim,num_layers,hidden_dim,class_num,
                 kernel_num,kernel_sizes,vocab_size,LSTM_dropout,CNN_dropout,Ci=1,
                 field=None,criteration="CrossEntropyLoss"):
        super(TC, self).__init__()

        V = vocab_size
        D = embed_dim
        C = class_num
        N = num_layers
        H = hidden_dim
        Ci = Ci
        Co = kernel_num
        Ks = kernel_sizes

        self.embed_A = nn.Embedding(V, D)
        self.embed_B = nn.Embedding(V, D)
        # self.embed_B.weight.data.copy_(field.vocab.vectors)
        self.classifier = TC_base(D,C,N,H,Ci,Co,Ks,LSTM_dropout,CNN_dropout)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()


    def forward(self, input_ids, labels, attention_mask=None,token_type_ids=None):
        x= input_ids
        x_A = self.embed_A(x)  # x [batch_size, sen_len, D]
        x_B = self.embed_B(x)
        x = torch.add(x_A, x_B)
        logits = self.classifier(x)
        loss = self.criteration(logits,labels)
        return loss,logits



class BERT_TC(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.bert_config = config
        self.embed_dim = config.hidden_size
        self.class_num = kwargs["class_num"]
        self.num_layers = kwargs["num_layers"]
        self.hidden_size = kwargs["hidden_dim"]
        self.Ci = kwargs["Ci"]
        self.kernel_num = kwargs["kernel_num"]
        self.kernel_sizes = kwargs["kernel_sizes"]
        self.LSTM_dropout = kwargs["LSTM_dropout"]
        self.CNN_dropout = kwargs["CNN_dropout"]

        self.bert = BertModel(self.bert_config)
        self.classifier = TC_base(self.embed_dim,self.class_num,self.num_layers,self.hidden_size,self.Ci,self.kernel_num,self.kernel_sizes,
                                  self.LSTM_dropout,self.CNN_dropout)

        if kwargs["criteration"] == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def extra_repr(self) -> str:
        return 'bert word embedding dim:{}'.format(
            self.embed_size
        )

    def forward(self, input_ids, labels, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        embedding = outputs[0]
        logits = self.classifier(embedding)
        loss = self.criteration(logits, labels)
        return loss, logits