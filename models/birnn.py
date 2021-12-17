import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel

class TC_base(nn.Module):
	def __init__(self, cell,  in_features, hidden_dim, num_layers, class_num, dropout_rate,):
		super(TC_base, self).__init__()
		self._cell = cell
		self.in_features=in_features
		self.class_num = class_num
		self.rnn = None
		if cell == 'rnn':
			self.rnn = nn.RNN(in_features, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = hidden_dim * num_layers
		elif cell == 'bi-rnn':
			self.rnn = nn.RNN(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'gru':
			self.rnn = nn.GRU(in_features, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = hidden_dim * num_layers
		elif cell == 'bi-gru':
			self.rnn = nn.GRU(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'lstm':
			self.rnn = nn.LSTM(in_features, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'bi-lstm':
			self.rnn = nn.LSTM(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 4 * hidden_dim * num_layers
		else:
			raise Exception("no such rnn cell")
		self.output_layer = nn.Linear(out_hidden_dim, class_num)

	def forward(self,features):
		_ = features.permute(1, 0, 2)
		__, h_out = self.rnn(_)
		if self._cell in ["lstm", "bi-lstm"]:
			h_out = torch.cat([h_out[0], h_out[1]], dim=2)
		h_out = h_out.permute(1, 0, 2)
		h_out = h_out.reshape(-1, h_out.shape[1] * h_out.shape[2])
		logits = self.output_layer(h_out)
		return logits

	def extra_repr(self) -> str:
		return 'features {}->{},'.format(
			self.in_features, self.class_num
		)


class TC(nn.Module):
	def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, class_num, dropout_rate, criteration="CrossEntropyLoss"):
		super(TC, self).__init__()
		self._cell = cell

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.classifier = TC_base(cell,embed_size,hidden_dim,num_layers,class_num,dropout_rate)
		if criteration == "CrossEntropyLoss":
			self.criteration = nn.CrossEntropyLoss()
		else:
			# default loss
			self.criteration = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels,attention_mask=None,token_type_ids=None):
		x = input_ids.long()
		embedding = self.embedding(x)
		logits = self.classifier(embedding)
		loss = self.criteration(logits, labels)
		return loss,logits

	def extra_repr(self) -> str:
		return 'features {}->{},'.format(
			self.embed_size, self.class_num
		)


class BERT_TC(BertPreTrainedModel):
	def __init__(self, config, **kwargs):
		super().__init__(config)
		self.class_num = kwargs["class_num"]
		self.dropout_rate = kwargs["dropout_rate"]
		self.cell = kwargs["cell"]
		self.hidden_dim = kwargs["hidden_dim"]
		self.num_layers = kwargs["num_layers"]
		self.embed_size = config.hidden_size  # not kwags["embed_size"]
		self.plm_config = config

		self.bert = BertModel(self.plm_config)
		self.classifier = TC_base(self.cell, self.embed_size, self.hidden_dim, self.num_layers, self.class_num, self.dropout_rate)
		if kwargs["criteration"] == "CrossEntropyLoss":
			self.criteration = nn.CrossEntropyLoss()
		else:
			# default loss
			self.criteration = nn.CrossEntropyLoss()


	def extra_repr(self) -> str:
		return 'bert word embedding dim:{}'.format(
			self.embed_size
		)


	def forward(self,input_ids, labels, attention_mask, token_type_ids):
		outputs = self.bert(input_ids, attention_mask, token_type_ids)
		embedding = outputs[0]
		logits = self.classifier(embedding)
		loss = self.criteration(logits, labels)
		return loss, logits
