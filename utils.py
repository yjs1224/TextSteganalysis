import json
import sklearn.metrics as metrics

class MyDict(dict):
    __setattr__ = dict.__setitem__
    # def __setattr__(self, key, value):
    #     try:
    #         self[key] = value
    #     except:
    #         raise  AttributeError(key)
    # __getattr__ = dict.__getitem__
    def __getattr__(self, item):
        try:
            return self[item]
        except:
            raise AttributeError(item)

class Config(object):
    def __init__(self, config_path):
        configs = json.load(open(config_path, "r", encoding="utf-8"))
        self.configs = self.dictobj2obj(configs)
        self.configs.state_dict = configs

    def dictobj2obj(self, dictobj):
        if not isinstance(dictobj, dict):
            return dictobj
        d = MyDict()
        for k, v in dictobj.items():
            d[k] = self.dictobj2obj(v)
        return d



    def get_configs(self):
        return self.configs


def compute_metrics(task_name, preds, labels, stego_label=1):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name in ["steganalysis", "graph_steganalysis"]:
        return {"accuracy": metrics.accuracy_score(labels, preds),
                "macro_f1":metrics.f1_score(labels, preds, average="macro"),
                "precision":metrics.precision_score(labels, preds, pos_label=stego_label),
                "recall":metrics.recall_score(labels, preds, pos_label=stego_label),
                "f1_score":metrics.f1_score(labels, preds, pos_label=stego_label)}
    else:
        raise KeyError(task_name)
