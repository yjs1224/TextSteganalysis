import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, get_att=False):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.get_att = get_att
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape  # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, out_features, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.hidden = n_hid
        self.max_length = 128
        self.dropout = 0.1
        self.attentions = [GATLayer(n_feat, n_hid, dropout=self.dropout, alpha=alpha, concat=True, get_att=False) for _
                           in range(n_heads)]
        # self.attentions_adj = [GATLayer(n_feat, self.max_length,  alpha=alpha, concat=True,get_att=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(n_hid * n_heads, out_features, dropout=self.dropout, alpha=alpha, concat=False)

    def forward(self, x_input, adj):
        x = F.dropout(x_input, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class TC(nn.Module):
    def __init__(self,vocab_size,clf, TC_configs, embed_size,class_num, hidden_dim, readout_size, gat_alpha, gat_heads,dropout_rate,strategy="cas", criteration="CrossEntropyLoss",):
        super(TC, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.clf_name = clf
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.hidden_dim = hidden_dim
        self.readout_size = readout_size
        self.gat_alpha = gat_alpha
        self.gat_heads = gat_heads
        self.strategy = strategy

        if self.clf_name == "cnn":
            from .cnn import TC_base
            self.clf_configs = TC_configs.cnn
        elif self.clf_name == "fc":
            from .fcn import TC_base
            self.clf_configs = TC_configs.fc
        elif self.clf_name == "rnn":
            from .birnn import TC_base
            self.clf_configs = TC_configs.rnn
        else:
            assert 0, "No such clf, only support cnn rnn & fc"
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.gat = GAT(
            n_feat=self.embed_size,
            n_hid=self.hidden_dim,
            out_features=self.readout_size,
            alpha=self.gat_alpha,
            n_heads=self.gat_heads,
            dropout=self.dropout_rate
        )
        if self.strategy.lower() == "cas":
            self.clf_configs.in_features = self.readout_size
        elif self.strategy.lower() == "parl":
            self.clf_configs.in_features = self.readout_size + self.embed_size
        self.classifier = TC_base(**{**self.clf_configs,"class_num":self.class_num,"dropout_rate":self.dropout_rate})
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

        # covert my dict to standard dict
        self.clf_configs = {**self.clf_configs}

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None, graph=None):
        embedding = self.embedding(input_ids)
        gat_out = self.gat(embedding, graph)

        if self.strategy.lower() == "cas":
            logits = self.classifier(gat_out)
        elif self.strategy.lower() == "parl":
            logits = self.classifier(torch.cat([gat_out,embedding],dim=2))
        loss = self.criteration(logits, labels)
        return loss, logits


class BERT_TC(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        TC_configs = kwargs["TC_configs"]
        self.embed_size = config.hidden_size
        self.clf_name = kwargs["clf"]
        self.dropout_rate = kwargs["dropout_rate"]
        self.class_num = kwargs["class_num"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.readout_size = kwargs["readout_size"]
        self.gat_alpha = kwargs["gat_alpha"]
        self.gat_heads = kwargs["gat_heads"]
        self.strategy = kwargs["strategy"]
        self.bert_config = config

        if self.clf_name == "cnn":
            from .cnn import TC_base
            self.clf_configs = TC_configs.cnn
        elif self.clf_name == "fc":
            from .fcn import TC_base
            self.clf_configs = TC_configs.fc
        elif self.clf_name == "rnn":
            from .birnn import TC_base
            self.clf_configs = TC_configs.rnn
        else:
            assert 0, "No such clf, only support cnn rnn & fc"

        self.bert = BertModel(config)
        self.gat = GAT(
            n_feat=self.embed_size,
            n_hid=self.hidden_dim,
            out_features=self.readout_size,
            alpha=self.gat_alpha,
            n_heads=self.gat_heads,
            dropout=self.dropout_rate
        )
        if self.strategy.lower() == "cas":
            self.clf_configs.in_features = self.readout_size
        elif self.strategy.lower() == "parl":
            self.clf_configs.in_features = self.readout_size + self.embed_size
        self.classifier = TC_base(
            **{**self.clf_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
        if kwargs["criteration"] == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                graph=None, labels=None,):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        embedding = outputs[0]  # [batch_size, node,hidden_size]
        gat_out = self.gat(embedding, graph)
        if self.strategy.lower() == "cas":
            logits = self.classifier(gat_out)
        elif self.strategy.lower() == "parl":
            logits = self.classifier(torch.cat([gat_out,embedding],dim=2))
        loss = self.criteration(logits, labels)
        return loss, logits
