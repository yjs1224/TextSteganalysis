import utils
import random
import dataset
import numpy as np
import logging
import os
import json
import time
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from models import birnn as BiRNN,cnn as CNN,lstmatt as LSTMATT,fcn as FCN, r_bilstm_c as RBC,\
    bilstm_dense as BLSTMDENSE,sesy as SESY


from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

task_metrics = {"steganalysis" : "accuracy",
                "graph_steganalysis" : "accuracy",}


logger = logging.getLogger(__name__)
time_stamp = "-".join(time.ctime().split())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(Configs, VOCAB_SIZE=None, checkpoint=None):
    # set model

    logger.info("----------------init model-----------------------")


    if not Configs.use_plm:
        if Configs.model.lower() in ["ts-csw", "cnn"]:
            Model_Configs = Configs.CNN
            Model_Configs.vocab_size = VOCAB_SIZE
            model = CNN.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["birnn"]:
            Model_Configs = Configs.RNN
            Model_Configs.vocab_size = VOCAB_SIZE
            model = BiRNN.TC(**{**Model_Configs, "class_num": Configs.class_num})
        elif Configs.model.lower() in ["fcn", "fc"]:
            Model_Configs = Configs.FCN
            Model_Configs.vocab_size = VOCAB_SIZE
            model = FCN.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["lstmatt"]:
            Model_Configs = Configs.LSTMATT
            Model_Configs.vocab_size = VOCAB_SIZE
            model = LSTMATT.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["r-bilstm-c", "r-b-c", "rbc", "rbilstmc"]:
            Model_Configs = Configs.RBiLSTMC
            Model_Configs.vocab_size = VOCAB_SIZE
            model = RBC.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["bilstmdense", "bilstm-dense", "bilstm_dense", "bi-lstm-dense"]:
            Model_Configs = Configs.BiLSTMDENSE
            Model_Configs.vocab_size = VOCAB_SIZE
            model = BLSTMDENSE.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["sesy"]:
            Model_Configs = Configs.SESY
            Model_Configs.vocab_size = VOCAB_SIZE
            model = SESY.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["gnn"]:
            Model_Configs = Configs.GNN
            Model_Configs.vocab_size = VOCAB_SIZE
            model = GNN.TC(**{**Model_Configs, "class_num": Configs.class_num, })
        else:
            logger.error("no such model, exit")
            exit()
        if checkpoint is not None:
            logger.info("---------------------loading model from {}------------\n\n".format(checkpoint))
            model = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
    else:
        if checkpoint is not None:
            logger.info("---------------------loading model from {}------------\n\n".format(checkpoint))
            model_name_or_path = checkpoint
        else:
            logger.info("-------------loading pretrained language model from huggingface--------------\n\n")
            model_name_or_path = Configs.model_name_or_path

        if Configs.model.lower() in ["ts-csw", "cnn"]:
            Model_Configs = Configs.CNN
            model = CNN.BERT_TC.from_pretrained(model_name_or_path,
                                                **{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["birnn"]:
            Model_Configs = Configs.RNN
            model = BiRNN.BERT_TC.from_pretrained(model_name_or_path,
                                                **{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["fcn", "fc"]:
            Model_Configs = Configs.FCN
            model = FCN.BERT_TC.from_pretrained(model_name_or_path,**{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["lstmatt"]:
            Model_Configs = Configs.LSTMATT
            model = LSTMATT.BERT_TC.from_pretrained(model_name_or_path,
                                                **{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["r-bilstm-c", "r-b-c", "rbc", "rbilstmc"]:
            Model_Configs = Configs.RBiLSTMC
            model = RBC.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["bilstmdense", "bilstm-dense", "bilstm_dense", "bi-lstm-dense"]:
            Model_Configs = Configs.BiLSTMDENSE
            model = BLSTMDENSE.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{**Model_Configs, "class_num": Configs.class_num, })
        elif Configs.model.lower() in ["sesy"]:
            Model_Configs = Configs.SESY
            model = SESY.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{**Model_Configs, "class_num": Configs.class_num, })
        else:
            logger.error("no such model, exit")
            exit()


    logger.info("Model Configs")
    logger.info(json.dumps({**{"MODEL_TYPE": Configs.model}, **Model_Configs, }))
    model = model.to(Configs.device)
    return model


def train_with_helper(data_helper,model,Configs,):
    os.makedirs(os.path.join(Configs.out_dir, Configs.checkpoint),exist_ok=True)
    checkpoint = os.path.join(Configs.out_dir, Configs.checkpoint, "maxacc.pth")
    Training_Configs = Configs.Training
    logger.info("Training Configs")
    logger.info(Training_Configs)
    logger.info("-----------------------------------------------")
    t_total = data_helper.train_num// Training_Configs.batch_size * Training_Configs.epoch
    num_warmup_steps = int(Training_Configs.warmup_ratio * t_total)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": Training_Configs.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Training_Configs.learning_rate, eps=Training_Configs.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    # optimizer = optim.SGD(model.parameters(), lr=Training_Configs.learning_rate, momentum=0.9)
    early_stop = 0
    best_acc = 0
    best_test_loss = 1000
    best_precison = 0
    best_recall = 0
    best_F1 = 0

    logger.info("------------number of instance-------------")
    logger.info(format(f"train\t{data_helper.train_num}"))
    logger.info(format(f"val  \t{data_helper.val_num}"))
    logger.info(format(f"test \t{data_helper.test_num}"))

    for epoch in range(Training_Configs.epoch):
        model.train()
        generator_train = data_helper.train_generator(Training_Configs.batch_size)
        train_loss = []
        train_acc = []
        while True:
            try:
                text, label = generator_train.__next__()
            except:
                break
            optimizer.zero_grad()
            loss,y = model(torch.from_numpy(text).long().to(Configs.device), torch.from_numpy(label).long().to(Configs.device))
            # loss = criteration(y, torch.from_numpy(label).long().to(Training_configs.device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            y = y.cpu().detach().numpy()
            train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        val_loss, val_acc, val_precision, val_recall, val_Fscore = eval_with_helper(data_helper,model,Configs)

        logger.info(
            "epoch {:d}, training loss {:.4f}, train acc {:.4f}, val loss {:.4f}, val acc {:.4f}, val pre {:.4f},val recall {:.4f},val F1 {:.4f}"
                .format(epoch + 1, np.mean(train_loss), np.mean(train_acc), val_loss, val_acc, val_precision, val_recall,
                        val_Fscore))


        if val_acc > best_acc:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "scheduler":scheduler.state_dict(),
                     "training_args":Configs.state_dict,"val loss": val_loss, "val acc": np.mean(val_acc)}
            torch.save(state, checkpoint)
            best_test_loss = val_loss
            best_acc = val_acc
            best_precison = val_precision
            best_recall = val_recall
            best_F1 = val_Fscore
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= Training_Configs.early_stop :
            break

    logger.info("--------------start calculate metrics--------------")

    state = torch.load(checkpoint)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    model.load_state_dict(state["model"])
    test_loss, test_acc, test_precision, test_recall, test_Fscore = eval_with_helper(data_helper,model,Configs,"test")
    logger.info('val: loss: {:.4f}, acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_test_loss, best_acc,
                                                                                              best_precison, best_recall, best_F1))
    logger.info(
        "test: loss {:.4f}, acc {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}".format(test_loss, test_acc, test_precision,
                                                                                     test_recall, test_Fscore))
    return test_acc, test_precision, test_recall, test_Fscore


def eval_with_helper(data_helper,model, Configs, eval_or_test="eval"):
    Training_Configs = Configs.Training
    model.eval()
    generator =  data_helper.val_generator(Training_Configs.batch_size) if eval_or_test == "eval" \
        else data_helper.test_generator(Training_Configs.batch_size)
    test_loss = 0
    test_acc = []
    test_tp = []
    tfn = []
    tpfn = []
    length_sum = 0

    while True:
        with torch.no_grad():
            try:
                text, label = generator.__next__()
            except:
                break
            loss,y = model(torch.from_numpy(text).long().to(Configs.device),
                            torch.from_numpy(label).long().to(Configs.device))
            # loss = criteration(y, torch.from_numpy(label).long().to(Training_configs.device))
            loss = loss.cpu().numpy()
            test_loss += loss * len(text)
            length_sum += len(text)
            y = y.cpu().numpy()
            label_pred = np.argmax(y, axis=-1)
            test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
            test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
            tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
            tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]

    test_loss = test_loss / length_sum
    acc = np.mean(test_acc)
    tpsum = np.sum(test_tp)
    test_precision = tpsum / (np.sum(tfn) + 1e-5)
    test_recall = tpsum / np.sum(tpfn)
    test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision + 1e-10)
    return test_loss, acc, test_precision, test_recall, test_Fscore


def train( model, Configs, tokenizer):
    train_dataset = load_and_cache_examples(Configs.Dataset, Configs.task_name, tokenizer)
    Training_Configs = Configs.Training_with_Processor
    Training_Configs.train_batch_size = Training_Configs.per_gpu_train_batch_size * max(1, Training_Configs.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=Training_Configs.train_batch_size)

    if Training_Configs.max_steps > 0:
        t_total = Training_Configs.max_steps
        Training_Configs.num_train_epochs = Training_Configs.max_steps // (len(train_dataloader) // Training_Configs.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // Training_Configs.gradient_accumulation_steps * Training_Configs.num_train_epochs

    num_warmup_steps = int(Training_Configs.warmup_ratio * t_total)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": Training_Configs.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Training_Configs.learning_rate, eps=Training_Configs.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(Training_Configs.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(Training_Configs.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(Training_Configs.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(Training_Configs.model_name_or_path, "scheduler.pt")))


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", Training_Configs.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", Training_Configs.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        Training_Configs.train_batch_size
        * Training_Configs.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", Training_Configs.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(Training_Configs.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(Training_Configs.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0

        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(epochs_trained, int(Training_Configs.num_train_epochs))

    set_seed(Configs.seed)  # Added here for reproductibility

    best_val_metric = None
    for epoch_n in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=False)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(Configs.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            if Configs.task_name == "graph_steganalysis":
                inputs = {**inputs,"graph":batch[4]}

            outputs = model(**inputs)
            loss = outputs[0]

            if Training_Configs.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if Training_Configs.gradient_accumulation_steps > 1:
                loss = loss / Training_Configs.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % Training_Configs.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Training_Configs.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                logs = {}
                if Training_Configs.logging_steps > 0 and global_step % Training_Configs.logging_steps == 0:
                    # Log metrics
                    if Training_Configs.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ , _ = evaluate( model, tokenizer, Configs, Configs.task_name, use_tqdm=False)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / Training_Configs.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["avg_loss_since_last_log"] = loss_scalar
                    logging_loss = tr_loss

                    logging.info(json.dumps({**logs, **{"step": global_step}}))


                if ( Training_Configs.eval_and_save_steps > 0 and global_step % Training_Configs.eval_and_save_steps == 0) \
                        or (step+1==t_total):
                    # evaluate
                    results, _, _ = evaluate(model, tokenizer, Configs, Configs.task_name, use_tqdm=False)
                    for key, value in results.items():
                        logs[f"eval_{key}"] = value
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # save
                    if Training_Configs.save_only_best:
                        output_dirs = [os.path.join(Configs.out_dir, Configs.checkpoint)]
                    else:
                        output_dirs = [os.path.join(Configs.out_dir, f"checkpoint-{global_step}")]
                    curr_val_metric = results[task_metrics[Configs.task_name]]
                    if best_val_metric is None or curr_val_metric > best_val_metric:
                        # check if best model so far
                        logger.info("Congratulations, best model so far!")
                        best_val_metric = curr_val_metric

                    for output_dir in output_dirs:
                        # in each dir, save model, tokenizer, args, optimizer, scheduler
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", output_dir)
                        if Configs.use_plm:
                            model_to_save.save_pretrained(output_dir)
                        else:
                            torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
                        torch.save(Configs.state_dict, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        tokenizer.save_pretrained(output_dir)
                        logger.info("\tSaved model checkpoint to %s", output_dir)


            if Training_Configs.max_steps > 0 and global_step > Training_Configs.max_steps:
                epoch_iterator.close()
                break
        if Training_Configs.max_steps > 0 and global_step > Training_Configs.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, Configs, task_name, split="dev", prefix="", use_tqdm=True):
    Training_Configs = Configs.Training_with_Processor
    results = {}
    if task_name == "record":
        eval_dataset, eval_answers = load_and_cache_examples(Configs.Dataset, task_name, tokenizer, split=split)
    else:
        eval_dataset = load_and_cache_examples(Configs.Dataset, task_name, tokenizer, split=split)

    if not os.path.exists(Configs.out_dir):
        os.makedirs(Configs.out_dir)

    Training_Configs.eval_batch_size = Training_Configs.per_gpu_eval_batch_size * max(1, Training_Configs.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=Training_Configs.eval_batch_size)

    # multi-gpu eval
    if Training_Configs.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", Training_Configs.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(Configs.device) for t in batch)
        guids = batch[-1]

        max_seq_length = batch[0].size(1)
        if Training_Configs.use_fixed_seq_length:  # no dynamic sequence length
            batch_seq_length = max_seq_length
        else:
            batch_seq_length = torch.max(batch[-2], 0)[0].item()

        if batch_seq_length < max_seq_length:
            inputs = {"input_ids": batch[0][:, :batch_seq_length].contiguous(),
                      "attention_mask": batch[1][:, :batch_seq_length].contiguous(),
                      "token_type_ids":batch[2][:, :batch_seq_length].contiguous(),
                      "labels": batch[3]}
            # inputs["token_type_ids"] = (
            #     batch[2][:, :batch_seq_length].contiguous() if Configs.model_type in ["bert", "xlnet", "albert"] else None
            # )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],  "token_type_ids":batch[2],"labels": batch[3]}

        if Configs.task_name == "graph_steganalysis":
            inputs = {**inputs,"graph":batch[4]}
            # inputs["token_type_ids"] = (
            #     batch[2][:, :batch_seq_length].contiguous() if Configs.model_type in ["bert", "xlnet", "albert"] else None
            # )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    ex_ids = np.concatenate(ex_ids, axis=0)
    eval_loss = eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)

    result = utils.compute_metrics(task_name, preds, out_label_ids,)
    results.update(result)
    if prefix == "":
        return results, preds, ex_ids
    output_eval_file = os.path.join(Configs.out_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** {split} results: {prefix} *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids


def load_and_cache_examples(Dataset_Configs, task, tokenizer, split="train"):
    if task == "steganalysis":
        from processors.process import SteganalysisProcessor as DataProcessor
    elif task == "graph_steganalysis":
        from processors.graph_process import GraphSteganalysisProcessor as DataProcessor

    processor  = DataProcessor(tokenizer)
    # Load data features from cache or dataset file
    cached_tensors_file = os.path.join(
        Dataset_Configs.csv_dir,
        "tensors_{}_{}_{}".format(
            split, time_stamp, str(task),
        ),
    )
    if os.path.exists(cached_tensors_file) and not Dataset_Configs.overwrite_cache:
        logger.info("Loading tensors from cached file %s", cached_tensors_file)
        start_time = time.time()
        dataset = torch.load(cached_tensors_file)
        logger.info("\tFinished loading tensors")
        logger.info(f"\tin {time.time() - start_time}s")

    else:
        # no cached tensors, process data from scratch
        logger.info("Creating features from dataset file at %s", Dataset_Configs.csv_dir)
        if split == "train":
            get_examples = processor.get_train_examples
        elif split == "dev":
            get_examples = processor.get_dev_examples
        elif split == "test":
            get_examples = processor.get_test_examples

        _,examples = get_examples(Dataset_Configs.csv_dir)
        dataset = processor.convert_examples_to_features(examples,)
        logger.info("\tFinished creating features")

        logger.info("\tFinished converting features into tensors")
        if Dataset_Configs.save_cache:
            logger.info("Saving features into cached file %s", cached_tensors_file)
            torch.save(dataset, cached_tensors_file)
            logger.info("\tFinished saving tensors")

    if task == "record" and split in ["dev", "test"]:
        answers = processor.get_answers(Dataset_Configs.csv_dir, split)
        return dataset, answers
    else:
        return dataset


def main(Configs):
    # args conflict checking
    if Configs.use_plm:
            assert Configs.use_processor, "\nWhen using plm, You can only use processor to process dataset!!\n"

    Dataset_Configs = Configs.Dataset
    Configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Configs.out_dir,exist_ok=True)
    set_seed(Configs.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    handler = logging.FileHandler(os.path.join(Configs.out_dir,time_stamp+"_log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("--------------Main Configs-------------------")
    logger.info(Configs)

    logger.info("--------------loading data-------------------")
    logger.info("Dataset Configs")
    logger.info(json.dumps(Dataset_Configs))

    # check whether to use plm
    logger.info("----------------use plm or not----------------")
    if Configs.use_plm:
        logger.info("------------------YES-----------------------------")
        Configs.model_name_or_path = Configs.Training_with_Processor.model_name_or_path
        logger.info("\tload plm name or path from Training_with_Processor args")
    else:
        logger.info("--------------------NO-------------------------")
        if Configs.use_processor:
            Configs.model_name_or_path = Configs.Training_with_Processor.model_name_or_path
            logger.info("\tload plm name or path from Training_with_Processor args")
        else:
            Configs.model_name_or_path = Configs.Tokenizer.model_name_or_path
            logger.info("\tload plm name or path from Tokenizer args")

    logger.info("-------------------------------------------------------------------------------------------------------")
    # prepare data
    if Configs.use_processor:
        # translate txt into csv
        if not Dataset_Configs.resplit and os.path.exists(Dataset_Configs.csv_dir) and \
                os.path.exists(os.path.join(Dataset_Configs.csv_dir,"train.csv")) and \
                os.path.exists(os.path.join(Dataset_Configs.csv_dir,"val.csv")) and \
                os.path.exists(os.path.join(Dataset_Configs.csv_dir,"val.csv")):
            pass
        else:
            os.makedirs(Dataset_Configs.csv_dir, exist_ok=True)
            with open(Dataset_Configs.cover_file, 'r', encoding='utf-8') as f:
                covers = f.read().split("\n")
            covers = list(filter(lambda x: x not in ['', None], covers))
            random.shuffle(covers)
            with open(Dataset_Configs.stego_file, 'r', encoding='utf-8') as f:
                stegos = f.read().split("\n")
            stegos = list(filter(lambda x: x not in ['', None],  stegos))
            random.shuffle(stegos)
            texts = covers+stegos
            labels = [0]*len(covers) + [1]*len(stegos)
            val_ratio = (1-Dataset_Configs.split_ratio)/Dataset_Configs.split_ratio
            train_texts,test_texts,train_labels,test_labels = train_test_split(texts,labels,train_size=Dataset_Configs.split_ratio)
            train_texts,val_texts,train_labels,val_labels = train_test_split(train_texts, train_labels, train_size=val_ratio)
            def write2file(X, Y, filename):
                with open(filename, "w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["text", "label"])
                    for x, y in zip(X, Y):
                        writer.writerow([x, y])
            write2file(train_texts,train_labels, os.path.join(Dataset_Configs.csv_dir,"train.csv"))
            write2file(val_texts, val_labels, os.path.join(Dataset_Configs.csv_dir, "val.csv"))
            write2file(test_texts, test_labels, os.path.join(Dataset_Configs.csv_dir, "test.csv"))
        tokenizer = AutoTokenizer.from_pretrained(Configs.model_name_or_path,)
        VOCAB_SIZE = tokenizer.vocab_size

    else:
        # not recommend
        with open(Dataset_Configs.cover_file, 'r', encoding='utf-8') as f:
            covers = f.read().split("\n")
        covers = list(filter(lambda x: x not in ['', None], covers))
        random.shuffle(covers)
        with open(Dataset_Configs.stego_file, 'r', encoding='utf-8') as f:
            stegos = f.read().split("\n")
        stegos = list(filter(lambda x: x not in ['', None],  stegos))
        random.shuffle( stegos)


        if Configs.tokenizer:
            Tokenizer_Configs = Configs.Tokenizer
            data_helper = dataset.BertDataHelper([covers,  stegos], ratio=Dataset_Configs.split_ratio,
                                                 tokenizer_config=Tokenizer_Configs)
        else:
            Vocabulary_Configs = Configs.Vocabulary
            data_helper = dataset.DataHelper([covers,  stegos], use_label=True,
                                             ratio=Dataset_Configs.split_ratio,
                                             word_drop=Vocabulary_Configs.word_drop,
                                             do_lower=Vocabulary_Configs.do_lower,
                                             max_length= Vocabulary_Configs.max_length)

        VOCAB_SIZE = data_helper.vocab_size

    model = load_model(Configs, VOCAB_SIZE=VOCAB_SIZE)

    logger.info("--------------start training--------------------")

    if Configs.use_processor:
        # train_dataset = load_and_cache_examples(Dataset_Configs, Configs.task_name, tokenizer)  # , evaluate=False)
        global_step, tr_loss = train(model, Configs, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        Tarining_Configs = Configs.Training_with_Processor

        checkpoints = [os.path.join(Configs.out_dir, Configs.checkpoint)]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=Tarining_Configs.do_lower_case)
            prefix = checkpoint.split("/")[-1]
            model = load_model(Configs, VOCAB_SIZE=tokenizer.vocab_size,checkpoint=checkpoint)
            # if not Configs.use_plm:
            #     model = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
            #     logger.info("--------------load model without pretrained language model-----------------")
            # else:
            #     logger.info("--------------load model with pretrained language model--------------------")
            result, preds, ex_ids = evaluate(model, tokenizer, Configs, Configs.task_name, split="test", prefix=prefix)
            test_acc = result["accuracy"]
            test_precision = result["precision"]
            test_recall = result["recall"]
            test_Fscore = result["f1_score"]

    else:
        test_acc, test_precision, test_recall, test_Fscore = train_with_helper(data_helper,model,Configs)

    record_file = Configs.record_file if Configs.record_file is not None else "record.txt"
    result_path = os.path.join(Configs.out_dir, time_stamp+"----"+record_file)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("test phase:\naccuracy\t{:.4f}\nprecision\t{:.4f}\nrecall\t{:.4f}\nf1_score\t{:.4f}"
                .format(test_acc*100,test_precision*100,test_recall*100,test_Fscore*100))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="argument for generation")
    parser.add_argument("--config_path", type=str, default="./configs/test.json")
    args = parser.parse_args()
    Configs = utils.Config(args.config_path).get_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = Configs.gpuid
    main(Configs)