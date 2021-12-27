import torch
import numpy as np
from torch.utils.data import TensorDataset
import csv
import os
import copy
import spacy
import stanza

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from stanza.server import CoreNLPClient
# corenlp = CoreNLPClient(annotators=["tokenize","ssplit","pos","lemma","depparse"],DEFAULT_ENDPOINT="http://localhost:9001")

# Corenlp = stanza.Pipeline(lang="en",processors='tokenize,pos,lemma,depparse',tokenize_pretokenized=True)

# def dependency_adj_matrix_test(text):
#     document = NLP(text)
#     seq_len = len(text.split())
#     matrix = np.zeros((seq_len, seq_len)).astype('float32')
#
#     for token in document:
#         if token.i < seq_len:
#             matrix[token.i][token.i] = 1
#             for child in token.children:
#                 if child.i < seq_len:
#                     matrix[token.i][child.i] = 1
#     return matrix


def dependency_adj_matrix(Corenlp, text, is_bidirectional=True,is_self_loop=True):
    # matrix_tmp = dependency_adj_matrix_test(text)
    # seq_len = len(text.split())
    # text ='the absence of violence and sex was refreshing'
    doc = Corenlp(text)
    words = []
    ids = []
    head_ids = []
    relations = []
    for sent in doc.sentences:
        for word in sent.words:
            ids.append(word.id)
            words.append(word.text)
            head_ids.append(word.head)
            relations.append(word.deprel)
    matrix_size = len(words)+1
    matrix = np.zeros((matrix_size,matrix_size))
    relation_mat = [["NULL" for _ in range(matrix_size)] for __ in range(matrix_size)]
    # relation_mat = np.zeros((matrix_size,matrix_size), dtype=str)
    for id, head_id, relation_type in zip(ids,head_ids, relations):
        matrix[id][head_id] = 1
        relation_mat[id][head_id] = relation_type
        if is_bidirectional:
            matrix[head_id][id] = 1
            relation_mat[head_id][id] = relation_type
        if is_self_loop:
            matrix[id][id] = 1
            relation_mat[id][id] = "selfloop"
    return matrix, np.array(relation_mat)


class InputExample(object):
    def __init__(self, sentence=None, label=None):
        self.sentence = sentence
        self.label = label
        # self.dependency = dependency


class SeqInputFeatures(object):
    def __init__(self, input_ids,input_mask,segment_ids, label_ids,dependency_adj):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.dependency_adj = dependency_adj


class GraphSteganalysisProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.order = 1
        self.max_seq_len = 128
        self.label_list = [0, 1]
        self.num_labels = 2
        self.label2id = {}
        self.id2label = {}
        self.cls_connect = True
        self.sep_connect = False
        self.use_stanza = True
        if self.use_stanza:
            self.tokenizer.add_special_tokens({"additional_special_tokens":['[unused0]']})
        for idx, label in enumerate(self.label_list):
            self.label2id[label] = idx
            self.id2label[idx] = label

    def get_examples(self, file_name):
        return self._create_examples(
            file_name=file_name
        )

    def get_train_examples(self, dir):
        return self.get_examples(os.path.join(dir, "train.csv"))

    def get_dev_examples(self, dir):
        return self.get_examples(os.path.join(dir, "val.csv"))

    def get_test_examples(self, dir):
        return self.get_examples(os.path.join(dir, "test.csv"))

    def _create_examples(self, file_name):
        examples = []
        file = file_name
        lines = csv.reader(open(file, 'r', encoding='utf-8'))
        # scores = []
        for i, line in enumerate(lines):
            if i > 0:
                sentence = line[0].lower().strip()
                label_t = line[1].strip()
                if label_t == "0":
                    label = 0
                if label_t == "1":
                    label = 1
                examples.append(InputExample(sentence=sentence, label=label))

        # dataset = self.convert_examples_to_features(examples)
        return  examples


    def merge(self, sentence, dependency):
        ## merge depedency of spacy format and bert format
        input_ids = self.tokenizer.encode_plus(sentence)["input_ids"][1:-1]
        padding_matrix_length = self.max_seq_len - len(input_ids)-2
        # new_dependency = np.zeros((len(input_ids),len(input_ids)))
        word_piece = [self.tokenizer.encode_plus(x)["input_ids"][1:-1] for x in sentence.split()]
        idx = 0
        dependency_list = dependency.tolist()
        for w_p in word_piece:
            if len(w_p) != 1:
                for row_idx in range(len(dependency_list)):
                    row = dependency_list[row_idx]
                    row = row[:idx]+ [row[idx]]*(len(w_p)-1) + row[idx:]
                    dependency_list[row_idx] = copy.deepcopy(row)
                dependency_list  = dependency_list[:idx]+ [dependency_list[idx]]*(len(w_p)-1)+dependency_list[idx:]
            idx += len(w_p)

        new_dependency = np.array(dependency_list)
        if new_dependency.shape[0] != new_dependency.shape[1]:
            print("error 1 sentence:%s"%sentence)
        if len(input_ids) != new_dependency.shape[0]:
            print("error 2 sentence: %s"%sentence)

        if padding_matrix_length <= 0:
            new_dependency = new_dependency[:self.max_seq_len-2, :self.max_seq_len-2]
            new_dependency = np.pad(new_dependency,((1,1),(1,1)), "constant")
            new_dependency[0,0]=1
            new_dependency[self.max_seq_len-1,self.max_seq_len-1]=1
        else:
            new_dependency = np.pad(new_dependency, ((1, padding_matrix_length+1), (1, padding_matrix_length+1)),
                                'constant')
            new_dependency[0,0]=1
            new_dependency[len(input_ids)+1,len(input_ids)+1]=1
        return new_dependency


    def convert_examples_to_features(self, examples):
        '''
        only for bert tokenizer
        '''

        Corenlp = stanza.Pipeline(lang="en", processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)
        features = []
        for example in examples:
            if self.use_stanza:
                # convert2stanzaformat
                # only for bert tokenizer
                inputs = self.tokenizer.encode_plus(
                    "[unused0] " + example.sentence,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation = True,
                    return_attention_mask=True,
                    return_token_type_ids=True
                )
            else:
                inputs = self.tokenizer.encode_plus(
                    example.sentence,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=True
                )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            if example.label is not None:
                label_id = self.label2id[example.label]
            else:
                label_id = -1

            # convert2stanzaformat
            # only for bert tokenizer
            if self.use_stanza:
                input_ids_tmp = self.tokenizer.encode_plus(example.sentence)["input_ids"]
                words = [self.tokenizer.decode(id) for id in input_ids_tmp[1:-1]]
                dependency, _ = dependency_adj_matrix(Corenlp, " ".join(words))
                # dependency, _ = dependency_adj_matrix(Corenlp=Corenlp, text=self.tokenizer.decode(input_ids_tmp[2:-1]))
                sep_position = len(input_ids_tmp)
            else:
                dependency, _ = dependency_adj_matrix(example.sentence)
            dependency_ = dependency
            for _ in range(self.order-1):
                dependency_ = np.matmul(dependency_, dependency)
            dependency_[dependency_>0] = 1
            # padding dependency
            padding_matrix_length = self.max_seq_len - dependency_.shape[0] - 2
            if padding_matrix_length <= 0:
                new_dependency = dependency_[:self.max_seq_len - 2, :self.max_seq_len - 2]
                constant_values = 1 if self.cls_connect else 0
                new_dependency = np.pad(new_dependency, ((1, 0),(1,0)), "constant",constant_values=constant_values)
                constant_values = 1 if self.sep_connect else 0
                new_dependency = np.pad(new_dependency, ((0,1), (0, 1)), "constant",constant_values=constant_values)
                new_dependency[0, 0] = 1
                new_dependency[self.max_seq_len - 1, self.max_seq_len - 1] = 1
            else:
                constant_values = 1 if self.cls_connect else 0
                new_dependency = np.pad(dependency_,
                                        ((1, 0), (1,0)),
                                        'constant',constant_values=constant_values)
                constant_values = 1 if self.sep_connect else 0
                new_dependency = np.pad(new_dependency,
                                        ((0,1), (0, 1)),
                                        'constant',constant_values=constant_values)
                constant_values = 0
                new_dependency = np.pad(new_dependency,
                                        ((0,padding_matrix_length), (0, padding_matrix_length)),
                                        'constant',constant_values=constant_values)
                new_dependency[0, 0] = 1
                new_dependency[sep_position, sep_position] = 1
            # return new_dependency
            # dependency = dependency_
            # dependency = self.merge(example.sentence,example.dependency)
            features.append(
            SeqInputFeatures(input_ids=input_ids,
                            input_mask=attention_mask,
                            segment_ids=token_type_ids,
                            label_ids=label_id,
                            dependency_adj=new_dependency))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_dependency_matrix = torch.tensor([f.dependency_adj for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dependency_matrix)
        return dataset


    def get_labels(self):
        return self.label_list


if __name__ == '__main__':
    '''
    function testing
    '''

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # processor = GraphSteganalysisProcessor(tokenizer)
    # get_examples = processor.get_train_examples
    # _, examples = get_examples('../data')
    # processor.get_test_examples("../data")
    # processor.get_dev_examples("../data")