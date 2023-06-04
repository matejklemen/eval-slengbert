import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Dict

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=128)
# SloBERTa: 79
parser.add_argument("--max_length", type=int, default=None)

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--eval_every_n_examples", type=int, default=5000)


if __name__ == "__main__":
	args = parser.parse_args()
	RANDOM_SEED = 17

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

	EVAL_EVERY_N_BATCHES = (args.eval_every_n_examples + args.batch_size - 1) // args.batch_size

	data = datasets.load_dataset("cjvt/sentinews", "sentence_level", split="train")
	# Mark the target feature
	data = data.class_encode_column("sentiment")
	data = data.rename_column("sentiment", "labels")  # most models expect targets inside `labels=...` argument
	num_ex = len(data)

	id2label = dict(enumerate(data.features["labels"].names))
	label2id = {_lbl: _i for _i, _lbl in id2label.items()}
	num_classes = len(id2label)

	rand_indices = np.random.permutation(num_ex)
	train_indices = rand_indices[:int(0.9 * num_ex)]
	dev_indices = rand_indices[int(0.9 * num_ex): int(0.95 * num_ex)]
	test_indices = rand_indices[int(0.95 * num_ex):]
	logging.info(f"Loaded {num_ex} examples: {len(train_indices)} train, {len(dev_indices)} dev, {len(test_indices)} test")

	# Save split indices for potential later comparisons
	with open(os.path.join(args.experiment_dir, "split_indices.json"), "w", encoding="utf-8") as f_split:
		json.dump({
			"train_indices": train_indices.tolist(),
			"dev_indices": dev_indices.tolist(),
			"test_indices": test_indices.tolist()
		}, fp=f_split, indent=4)

	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)

	# TODO: uncomment this to find a reasonable max length to pad sequences to
	# tmp_encoded = tokenizer.batch_encode_plus(data.select(train_indices)["content"])
	# tmp_lengths = sorted([len(_curr) for _curr in tmp_encoded["input_ids"]])
	# print(f"95th perc.: {tmp_lengths[int(0.95 * len(tmp_lengths))]}")
	# print(f"99th perc.: {tmp_lengths[int(0.99 * len(tmp_lengths))]}")
	# exit(0)
	# TODO: ----

	def tokenize_function(examples):
		return tokenizer(examples["content"], padding="max_length", max_length=args.max_length, truncation=True)

	tokenized_data = data.map(tokenize_function, batched=True)
	train_data = tokenized_data.select(train_indices)
	dev_data = tokenized_data.select(dev_indices)
	test_data = tokenized_data.select(test_indices)

	train_distribution = {id2label[_lbl_int]: _count / len(train_data)
						  for _lbl_int, _count in Counter(train_data['labels']).most_common()}
	logging.info(f"Training distribution: {train_distribution}")

	model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
															   num_labels=num_classes,
															   id2label=id2label, label2id=label2id)

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
		"text": test_data["content"],
		"pred_probas": pred_probas.tolist(),
		"pred_class": list(map(lambda _lbl_int: id2label[_lbl_int], pred_class.tolist()))
	})
	test_res.to_json(os.path.join(args.experiment_dir, "test_preds.json"), orient="records", lines=True)














