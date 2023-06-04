import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from typing import Dict

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction


# https://gist.github.com/gruber/8891611
URL_REGEX = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"


def preprocess_text(s):
    _s = re.sub(r"@(\w|\d)+", "[MENTION]", s)  # mentions
    _s = re.sub(URL_REGEX, "", _s)  # urls
    _s = re.sub("(“|‘|'|«|»|„|”)", "\"", _s)  # quotations
    _s = _s.strip()

    return _s


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=None)

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--eval_every_n_examples", type=int, default=5000)


if __name__ == "__main__":
    args = parser.parse_args()
    RANDOM_SEED = 17
    IDX2CLASS = {
        0: "not_offensive",
        1: "offensive"
    }

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(args.experiment_dir, exist_ok=True)
    ts = time.time()
    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, f"train{ts}.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    if args.max_length is None:
        args.max_length = 64
        logging.warning("--max_length is not set. Using max_length=64 as default, but you should set this to a "
                        "reasonable number such as the 95th or 99th percentile of training sequence lengths")

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    EVAL_EVERY_N_BATCHES = (args.eval_every_n_examples + args.batch_size - 1) // args.batch_size

    data = datasets.load_dataset("classla/FRENK-hate-sl", "binary")
    train_data = data["train"]
    train_data = train_data.to_pandas()
    train_data["text"] = train_data["text"].apply(preprocess_text)
    train_data["label"] = train_data["label"].apply(lambda _i: IDX2CLASS[_i])
    train_data = Dataset.from_pandas(train_data)

    dev_data = data["validation"]
    dev_data = dev_data.to_pandas()
    dev_data["text"] = dev_data["text"].apply(preprocess_text)
    dev_data["label"] = dev_data["label"].apply(lambda _i: IDX2CLASS[_i])
    dev_data = Dataset.from_pandas(dev_data)

    test_data = data["test"]
    test_data = test_data.to_pandas()
    test_data["text"] = test_data["text"].apply(preprocess_text)
    test_data["label"] = test_data["label"].apply(lambda _i: IDX2CLASS[_i])
    test_data = Dataset.from_pandas(test_data)

    data = datasets.concatenate_datasets([train_data, dev_data, test_data])

    # Mark the target feature
    data = data.class_encode_column("label")
    data = data.rename_column("label", "labels")  # most models expect targets inside `labels=...` argument
    num_ex = len(data)

    id2label = dict(enumerate(data.features["labels"].names))
    label2id = {_lbl: _i for _i, _lbl in id2label.items()}
    num_classes = len(id2label)

    train_data = Dataset.from_dict(data[:len(train_data)])
    dev_data = Dataset.from_dict(data[len(train_data): len(train_data) + len(dev_data)])
    test_data = Dataset.from_dict(data[-len(test_data):])

    logging.info(f"Loaded {num_ex} examples: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MENTION]"]})

    # TODO: uncomment this to find a reasonable max length to pad sequences to
    # tmp_encoded = tokenizer.batch_encode_plus(train_data["text"])
    # tmp_lengths = sorted([len(_curr) for _curr in tmp_encoded["input_ids"]])
    # print(f"95th perc.: {tmp_lengths[int(0.95 * len(tmp_lengths))]}")
    # print(f"99th perc.: {tmp_lengths[int(0.99 * len(tmp_lengths))]}")
    # exit(0)
    # TODO: ----

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=args.max_length, truncation=True)

    tokenized_data = data.map(tokenize_function, batched=True)
    train_data = train_data.map(tokenize_function, batched=True)
    dev_data = dev_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    train_distribution = {id2label[_lbl_int]: _count / len(train_data)
                          for _lbl_int, _count in Counter(train_data['labels']).most_common()}
    logging.info(f"Training distribution: {train_distribution}")

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
                                                               num_labels=num_classes,
                                                               id2label=id2label, label2id=label2id)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.experiment_dir,
        do_train=True, do_eval=True, do_predict=True,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps", logging_steps=EVAL_EVERY_N_BATCHES,
        save_strategy="steps", save_steps=EVAL_EVERY_N_BATCHES, save_total_limit=1,
        seed=RANDOM_SEED, data_seed=RANDOM_SEED,
        evaluation_strategy="steps", eval_steps=EVAL_EVERY_N_BATCHES,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        optim="adamw_torch",
        report_to="none",
    )
    accuracy_func = evaluate.load("accuracy")
    precision_func = evaluate.load("precision")
    recall_func = evaluate.load("recall")
    f1_func = evaluate.load("f1")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, int]:
        pred_logits, ground_truth = eval_pred
        predictions = np.argmax(pred_logits, axis=-1)

        metrics = {}
        metrics.update(accuracy_func.compute(predictions=predictions, references=ground_truth))
        macro_f1 = 0.0
        for _lbl_int, _lbl_str in id2label.items():
            bin_preds = (predictions == _lbl_int).astype(np.int32)
            bin_ground_truth = (ground_truth == _lbl_int).astype(np.int32)

            curr = precision_func.compute(predictions=bin_preds, references=bin_ground_truth)
            curr[f"precision_{_lbl_str}"] = curr.pop("precision")
            curr.update(recall_func.compute(predictions=bin_preds, references=bin_ground_truth))
            curr[f"recall_{_lbl_str}"] = curr.pop("recall")
            curr.update(f1_func.compute(predictions=bin_preds, references=bin_ground_truth))
            curr[f"f1_{_lbl_str}"] = curr.pop("f1")
            macro_f1 += curr[f"f1_{_lbl_str}"]

            metrics.update(curr)

        metrics["f1_macro"] = macro_f1 / max(1, len(id2label))

        return metrics

    trainer = Trainer(
        model=model, args=training_args, tokenizer=tokenizer,
        train_dataset=train_data, eval_dataset=dev_data,
        compute_metrics=compute_metrics
    )

    train_metrics = trainer.train()

    test_metrics = trainer.predict(test_dataset=test_data)
    logging.info(test_metrics.metrics)
    pred_probas = torch.softmax(torch.from_numpy(test_metrics.predictions), dim=-1)
    pred_class = torch.argmax(pred_probas, dim=-1)
    test_res = pd.DataFrame({
        "text": test_data["text"],
        "pred_probas": pred_probas.tolist(),
        "pred_class": list(map(lambda _lbl_int: id2label[_lbl_int], pred_class.tolist()))
    })
    test_res.to_json(os.path.join(args.experiment_dir, "test_preds.json"), orient="records", lines=True)














