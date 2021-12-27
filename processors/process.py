import torch
from torch.utils.data import TensorDataset
import csv
import os
import json


class InputExample(object):
    def __init__(self, sentence=None, label=None):
        self.sentence = sentence
        self.label = label


class SeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class SteganalysisProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_len = 128
        self.label_list = [0, 1]
        self.num_labels = 2
        self.label2id = {}
        self.id2label = {}
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
        # return dataset, examples
        return examples


    def convert_examples_to_features(self, examples):
        features = []
        for example in examples:
            inputs = self.tokenizer.encode_plus(
                example.sentence,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
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

            features.append(
                SeqInputFeatures(input_ids=input_ids,
                                 input_mask=attention_mask,
                                 segment_ids=token_type_ids,
                                 label_ids=label_id,))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
        # return features

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "steganalysis.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")

    def get_labels(self):
        return self.label_list