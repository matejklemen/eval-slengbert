import argparse
import json
import logging
import os
import sys
import time
from typing import List

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer

import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")

parser.add_argument("--train_path", type=str, default="janes_tag-preprocessed/fold0/train.jsonl")
parser.add_argument("--dev_path", type=str, default="janes_tag-preprocessed/fold0/dev.jsonl")
parser.add_argument("--test_path", type=str, default="janes_tag-preprocessed/fold0/test.jsonl")
parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--eval_every_n_examples", type=int, default=5000)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings and possible deadlocks

    args = parser.parse_args()
    RANDOM_SEED = 42

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

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    EVAL_EVERY_N_BATCHES = (args.eval_every_n_examples + args.batch_size - 1) // args.batch_size
    train_data = datasets.load_dataset("json", data_files=args.train_path, split="train")
    dev_data = datasets.load_dataset("json", data_files=args.dev_path, split="train")
    test_data = datasets.load_dataset("json", data_files=args.test_path, split="train")

    # Note: assuming later on that B-* labels are on indices 1, 3, 5, ...
    uniq_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
    id2label = {_i: _lbl for _i, _lbl in enumerate(uniq_labels)}
    label2id = {_lbl: _i for _i, _lbl in id2label.items()}
    num_classes = len(id2label)

    train_data = train_data.to_pandas()
    train_data["nes"] = train_data["nes"].apply(lambda _ne_tags: list(map(lambda _lbl: label2id[_lbl], _ne_tags)))
    train_data = Dataset.from_pandas(train_data)

    dev_data = dev_data.to_pandas()
    dev_data["nes"] = dev_data["nes"].apply(lambda _ne_tags: list(map(lambda _lbl: label2id[_lbl], _ne_tags)))
    dev_data = Dataset.from_pandas(dev_data)

    test_data = test_data.to_pandas()
    test_data["nes"] = test_data["nes"].apply(lambda _ne_tags: list(map(lambda _lbl: label2id[_lbl], _ne_tags)))
    test_data = Dataset.from_pandas(test_data)

    logging.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test examples")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MENTION]", "[URL]"]})

    def align_labels_with_tokens(labels, word_ids, loss_ignore_index=-100):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = loss_ignore_index if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(loss_ignore_index)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
        all_labels = examples["nes"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs


    enc_train_data = train_data.map(tokenize_and_align_labels, batched=True, remove_columns=train_data.column_names)
    enc_dev_data = dev_data.map(tokenize_and_align_labels, batched=True, remove_columns=dev_data.column_names)
    enc_test_data = test_data.map(tokenize_and_align_labels, batched=True, remove_columns=test_data.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[_l] for _l in label if _l != -100] for label in labels]
        true_predictions = [
            [id2label[_p] for (_p, _l) in zip(prediction, label) if _l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

        return all_metrics


    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_name_or_path,
                                                            id2label=id2label, label2id=label2id)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.experiment_dir,
        do_train=True, do_eval=True, do_predict=True,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_strategy="steps", logging_steps=EVAL_EVERY_N_BATCHES,
        save_strategy="steps", save_steps=EVAL_EVERY_N_BATCHES, save_total_limit=1,
        seed=RANDOM_SEED, data_seed=RANDOM_SEED,
        evaluation_strategy="steps", eval_steps=EVAL_EVERY_N_BATCHES,
        load_best_model_at_end=True, metric_for_best_model="overall_f1", greater_is_better=True,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=enc_train_data,
        eval_dataset=enc_dev_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

    test_metrics = trainer.predict(test_dataset=enc_test_data)
    logging.info(test_metrics.metrics)

    test_logits = torch.from_numpy(test_metrics.predictions)  # num_ex, max_seq_length_in_data, num_labels
    test_probas = torch.softmax(test_logits, dim=-1)
    test_preds = torch.argmax(test_probas, dim=-1)

    _test_enc = tokenizer(test_data["words"], is_split_into_words=True, max_length=test_logits.shape[1], padding="max_length")
    test_word_preds, test_word_probas = [], []
    for idx_ex in range(len(test_data)):
        word_ids = _test_enc.word_ids(idx_ex)
        curr_preds = test_preds[idx_ex]
        curr_probas = test_probas[idx_ex]

        # Note: word_ids are 0-based, so there are (1 + max_word_id) words in the example
        max_word_id = max(filter(lambda _w_id: _w_id is not None, word_ids))
        word_preds = [[] for _ in range(1 + max_word_id)]
        word_probas = [[] for _ in range(1 + max_word_id)]
        for idx_subw, idx_w in enumerate(word_ids):
            if idx_w is None:
                continue

            word_preds[idx_w].append(int(curr_preds[idx_subw]))
            word_probas[idx_w].append(float(curr_probas[idx_subw, curr_preds[idx_subw]]))

        # TODO: aggregation strategy? - currently: take prediction for first subword of the word
        test_word_preds.append([id2label[_preds[0]] for _preds in word_preds])
        test_word_probas.append([_probas[0] for _probas in word_probas])
        assert len(test_word_preds[idx_ex]) == len(test_data["nes"][idx_ex])

    correct_word_tags: List[List[str]] = list(
        map(lambda ne_tags: list(map(lambda _idx_class: id2label[_idx_class], ne_tags)), test_data["nes"])
    )
    test_metrics = metric.compute(predictions=test_word_preds,
                                  references=correct_word_tags)
    logging.info(test_metrics)

    test_res = pd.DataFrame({
        "text": test_data["words"],
        "pred_probas": test_word_probas,
        "pred_class": test_word_preds
    })
    test_res.to_json(os.path.join(args.experiment_dir, "test_preds.json"), orient="records", lines=True)

