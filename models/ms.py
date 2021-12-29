import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel
import os

class MyBert(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.bert_config = config
        self.bert = BertModel(self.bert_config)
        self.class_num = kwargs["class_num"]
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.class_num)
        if kwargs["criteration"] == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(outputs[1])
        loss = self.criteration(logits, labels)
        return loss, logits


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, filter_num, class_num, dropout_rate, criteration="CrossEntropyLoss",):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.class_num = class_num
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.conv = nn.Conv2d(1, filter_num, (3,self.embed_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(filter_num, self.class_num)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, attention_mask=None,token_type_ids=None):
        clf_input = self.embedding(input_ids).unsqueeze(3).permute(0,3,1,2)
        clf_input = self.conv(clf_input)
        clf_input = nn.functional.relu(clf_input).squeeze(3)
        clf_input = nn.functional.max_pool1d(clf_input, clf_input.size(2)).squeeze(2)
        clf_input = self.dropout(clf_input)
        logits = self.classifier(clf_input)
        loss = self.criteration(logits, labels)
        return loss, logits


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,num_layers, bidirectional, class_num, criteration="CrossEntropyLoss", ):
        super(BiRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=num_layers,bidirectional=bidirectional,batch_first=True )
        self.classifier = nn.Linear(2*self.hidden_size, self.class_num)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        clf_input = self.embedding(input_ids)
        clf_input, _ = self.lstm(clf_input)
        logits = self.classifier(clf_input[:,-1,:])
        loss = self.criteration(logits, labels)
        return loss, logits


class BiRNN_C(nn.Module):
    def __init__(self, class_num, dropout_rate, criteration="CrossEntropyLoss", **kwargs):
        super(BiRNN_C, self).__init__()
        cnn = torch.load(os.path.join(kwargs["cnn_checkpoint"], "pytorch_model.bin"))
        rnn = torch.load(os.path.join(kwargs["birnn_checkpoint"], "pytorch_model.bin"))
        # self.vocab_size = vocab_size
        # self.embed_size = embed_size
        # self.hidden_size = hidden_size
        self.class_num = class_num
        self.embed_C = cnn.embedding
        self.embed_R = rnn.embedding
        self.lstm = rnn.lstm
        self.conv = cnn.conv
        # self.embed_C = nn.Embedding(self.vocab_size, self.embed_size)
        # self.embed_R = nn.Embedding(self.vocab_size, self.embed_size)
        # self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=num_layers,bidirectional=bidirectional,batch_first=True)
        # self.conv = nn.Conv2d(1, filter_num, (3, self.embed_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(cnn.classifier.in_features+rnn.classifier.in_features, self.class_num)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        rnn_input = self.embed_R(input_ids)
        rnn_input, _ = self.lstm(rnn_input)
        rnn_input = rnn_input[:,-1,:]

        cnn_input = self.embed_C(input_ids).unsqueeze(3).permute(0,3,1,2)
        cnn_input = self.conv(cnn_input)
        cnn_input = nn.functional.relu(cnn_input).squeeze(3)
        cnn_input = nn.functional.max_pool1d(cnn_input, cnn_input.size(2)).squeeze(2)

        clf_input = torch.cat([cnn_input, rnn_input],dim=1)
        clf_input = self.dropout(clf_input)
        logits = self.classifier(clf_input)
        loss = self.criteration(logits, labels)
        return loss, logits