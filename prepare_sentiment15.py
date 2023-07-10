# Twitter sentiment for 15 European languages
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

# https://gist.github.com/gruber/8891611
URL_REGEX = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"


def preprocess_text(s):
	_s = re.sub(r"@(\w|\d)+", "[MENTION]", s)  # mentions
	_s = re.sub(URL_REGEX, "", _s)  # urls
	_s = re.sub("(“|‘|'|«|»|„|”)", "\"", _s)  # quotations
	_s = _s.strip()

	return _s


if __name__ == "__main__":
	np.random.seed(17)
	data_dir = "/home/matej/Documents/data/mozetic_sentiment"
	label_dir = os.path.join(data_dir, "TweetText_Label")

	lang = "slovenian"
	####################

	lang = f"{lang[0].upper()}{lang[1:]}"

	df = pd.read_csv(os.path.join(label_dir, f"{lang}_tweet_label.csv"))
	print(f"Initial examples: {df.shape[0]}")

	keep_indices, aggr_label = [], []
	for tweet_text, curr_group in df.groupby("Tweet text"):
		first_idx = curr_group.index.tolist()[0]
		majority_class = curr_group.iloc[0]["SentLabel"]

		if curr_group.shape[0] > 1:
			c = Counter(curr_group["SentLabel"].tolist())
			if len(c) > 1:
				sorted_counts = c.most_common()
				if sorted_counts[0][1] == sorted_counts[1][1]:
					continue  # drop

				majority_class = sorted_counts[0][0]
			else:
				majority_class = curr_group.iloc[0]["SentLabel"]

			first_idx = curr_group.index.tolist()[0]

		keep_indices.append(first_idx)
		aggr_label.append(majority_class)

	df.loc[keep_indices, "SentLabel"] = aggr_label
	df = df.iloc[keep_indices].reset_index(drop=True)

	df["Tweet text"] = df["Tweet text"].apply(preprocess_text)
	df = df.rename(columns={"Tweet text": "content", "SentLabel": "sentiment"})
	rand_indices = np.random.permutation(df.shape[0])

	df_test = df.iloc[rand_indices[-10_000:]]
	df_dev = df.iloc[rand_indices[-15_000: -10_000]]
	df_train = df.iloc[rand_indices[:-15_000]]

	print(f"{df_train.shape[0]} training, "
		  f"{df_dev.shape[0]} validation, "
		  f"{df_test.shape[0]} test examples")

	TARGET_DATA_DIR = "sentiment15-preprocessed"
	os.makedirs(TARGET_DATA_DIR, exist_ok=True)
	df_train.to_csv(os.path.join(TARGET_DATA_DIR, "train.csv"), sep=",", index=False)
	df_dev.to_csv(os.path.join(TARGET_DATA_DIR, "dev.csv"), sep=",", index=False)
	df_test.to_csv(os.path.join(TARGET_DATA_DIR, "test.csv"), sep=",", index=False)

















