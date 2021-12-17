import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertModel

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(lstm, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.LNx = nn.LayerNorm(4 * self.hidden_size)
        self.LNh = nn.LayerNorm(4 * self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4 * self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4 * self.hidden_size, bias=True)

    def forward(self, x):
        def recurrence(xt, hidden):  # enhanced with layer norm
            # input: input to the current cell
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))
            return ht, ct

        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        inputs = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(inputs[t], hidden)
            output.append(hidden[0])
        output = torch.stack(output, 0).transpose(0, 1)
        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(inputs[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


class TC_base(nn.Module):
    def __init__(self,in_features, hidden_dim,  class_num, dropout_rate,bidirectional):
        super(TC_base, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num
        self.hidden_size = hidden_dim
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm = lstm(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.attn = Attention(
            embed_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            n_head=1,
            score_function='mlp',
            dropout=self.dropout_prob
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, mask, input_ids_len):
        output = self.lstm(features)
        output = self.dropout(output)
        scc, scc1 = self.attn(output,output)
        t = input_ids_len.view( input_ids_len.size(0),1)
        scc_sen = torch.sum(scc,dim=2)
        scc_mean = torch.div(torch.sum(scc,dim=1),t)
        logits = self.classifier(scc_mean)
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )


class TC(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim,  class_num, dropout_rate,bidirectional=True,criteration="CrossEntropyLoss"):
        super(TC,self).__init__()
        self.embed_size = embed_size
        self.dropout_prob = dropout_rate
        self.num_labels = class_num
        self.bidirectional = bidirectional
        self.hidden_size = hidden_dim
        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.classifier = TC_base(self.embed_size,self.hidden_size,self.num_labels,self.dropout_prob,self.bidirectional)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()
        # self.it_weights()


    def forward(self, input_ids,labels,attention_mask=None,token_type_ids=None):
        input_ids_len = torch.sum(input_ids != 0, dim=-1).float()
        input_lstm = self.embed(input_ids.long())[0]
        mask = torch.ones_like(input_ids.long())
        mask[input_ids.long() != 0 ] = 0
        logits = self.classifier(input_lstm,mask,input_ids_len)
        loss = self.criteration(logits,labels)
        return loss,logits


class BERT_TC(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.bert_config = config
        self.bert = BertModel(self.bert_config)
        self.embed_size = config.hidden_size
        self.hidden_size = kwargs["hidden_dim"]
        self.num_labels = kwargs["class_num"]
        self.dropout_prob = kwargs["dropout_rate"]
        self.bidirectional = kwargs["bidirectional"]
        self.classifier = TC_base(self.embed_size, self.hidden_size, self.num_labels, self.dropout_prob,
                                  self.bidirectional)
        if kwargs["criteration"] == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()
        # self.it_weights()

    def forward(self,input_ids, labels, attention_mask, token_type_ids):
        input_ids_len = torch.sum(input_ids != 0, dim=-1).float()
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        embedding = outputs[0]
        mask = torch.ones_like(input_ids.long())
        mask[input_ids.long() != 0 ] = 0
        logits = self.classifier(embedding,mask,input_ids_len)
        loss = self.criteration(logits,labels)
        return loss,logits


    def extra_repr(self) -> str:
        return 'bert word embedding dim:{}'.format(
            self.embed_size
        )