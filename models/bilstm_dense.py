import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class TC_base(nn.Module):
    def __init__(self, in_features,class_num, num_layers,hidden_size,dropout_rate):
        super(TC_base, self).__init__()
        self.in_features= in_features
        self.class_num = class_num

        D = in_features
        C = class_num
        N = num_layers
        H = hidden_size

        self.lstm1 = nn.LSTM(D, H, num_layers=N, \
                             bidirectional=True,
                             batch_first=True,dropout=dropout_rate)

        self.lstm2 = nn.LSTM(2 * H, H, num_layers=N, \
                             bidirectional=True,
                             batch_first=True,dropout=dropout_rate)

        self.lstm3 = nn.LSTM(4 * H, H, num_layers=N, \
                             bidirectional=True,
                             batch_first=True,dropout=dropout_rate)

        self.fc1 = nn.Linear(2 * H, C)

    def forward(self, features):
        out1, _ = self.lstm1(features)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(torch.cat([out1, out2], 2))
        out = torch.add(torch.add(out1, out2), out3)
        logits = self.fc1(out[:, -1, :])
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )


class TC(nn.Module):
    def __init__(self,  vocab_size, embed_dim,class_num, num_layers,hidden_dim,dropout_rate,criteration="CrossEntropyLoss"):
        super(TC, self).__init__()

        V = vocab_size
        D = embed_dim
        C = class_num
        N = num_layers
        H = hidden_dim

        self.embed = nn.Embedding(V, D)
        # self.classifier = TC_base(embed_dim,class_num,num_layers,hidden_size,dropout_rate)
        self.classifier = TC_base(D,C, N, H, dropout_rate)

        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def forward(self, input_ids,labels,attention_mask=None,token_type_ids=None):
        embedding = self.embed(input_ids)
        logits = self.classifier(embedding)
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
        self.dropout_rate = kwargs["dropout_rate"]

        self.bert = BertModel(self.bert_config)
        self.classifier = TC_base(self.embed_dim,self.class_num,self.num_layers,self.hidden_size,self.dropout_rate)
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
