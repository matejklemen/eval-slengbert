import os
import re

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
    """ 
        Script to prepare IMSYPP-sl dataset:
        - keeps comments that have an agreement in annotation (either single annotation or both annotators agree),
        - removes a few comments that don't have an annotation (because it was not a Slovene tweet),
        - maps 4-label classification problem into a binary one
        - sets aside 10% of training set as a validation set
        - writes data to train/dev/test.csv
    """
    np.random.seed(17)
    train_df = pd.read_csv("/home/matej/Documents/data/imsypp/IMSyPP_SI_anotacije_round1-no_conflicts.csv")
    train_df["besedilo"] = train_df["besedilo"].apply(preprocess_text)
    test_df = pd.read_csv("/home/matej/Documents/data/imsypp/IMSyPP_SI_anotacije_round2.csv")
    test_df["besedilo"] = test_df["besedilo"].apply(preprocess_text)
    LABEL_MAP = {
        "0 ni sporni govor": "not_offensive", "1 nespodobni govor": "offensive", "2 žalitev": "offensive", "3 nasilje": "offensive"
    }

    # Each comment is annotated by one or two annotators: keep only those for which annotations agree
    # Also map labels to binary ones (not_offensive/offensive)
    train_comments, train_labels = [], []
    num_train_skipped = 0
    for tweet, group in train_df.groupby("ID"):
        labels = set(group["vrsta"].values)
        if len(labels) == 1:
            curr_label = labels.pop()
            if curr_label not in LABEL_MAP:
                print(f"Skipping example because label is '{curr_label}'")
                continue

            train_comments.append(group.iloc[0]["besedilo"])
            train_labels.append(LABEL_MAP[curr_label])
        else:
            num_train_skipped += 1

    print(f"Skipped {num_train_skipped} training examples due to disagreement")
    dedup = pd.DataFrame({"content": train_comments, "label": train_labels})
    indices = np.random.permutation(dedup.shape[0])
    bnd = int(0.9 * dedup.shape[0])
    train_indices, dev_indices = indices[:bnd], indices[bnd:]

    train_df = dedup.iloc[train_indices]
    dev_df = dedup.iloc[dev_indices]

    print(f"Training distribution ({train_df.shape[0]} examples):")
    print(train_df["label"].value_counts(normalize=True))
    print(f"Validation distribution ({dev_df.shape[0]} examples):")
    print(dev_df["label"].value_counts(normalize=True))

    TARGET_DATA_DIR = "imsypp-preprocessed"
    os.makedirs(TARGET_DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(TARGET_DATA_DIR, "train.csv"), sep=",", index=False)
    dev_df.to_csv(os.path.join(TARGET_DATA_DIR, "dev.csv"), sep=",", index=False)

    test_comments, test_labels = [], []
    num_test_skipped = 0
    for tweet, group in test_df.groupby("ID"):
        labels = set(group["vrsta"].values)
        if len(labels) == 1:
            curr_label = labels.pop()
            if curr_label not in LABEL_MAP:
                print(f"Skipping example because label is '{curr_label}'")
                continue

            test_comments.append(group.iloc[0]["besedilo"])
            test_labels.append(LABEL_MAP[curr_label])
        else:
            num_test_skipped += 1

    test_df = pd.DataFrame({"content": test_comments, "label": test_labels})
    print(f"Test distribution ({test_df.shape[0]} examples):")
    print(test_df["label"].value_counts(normalize=True))
    test_df.to_csv(os.path.join(TARGET_DATA_DIR, "test.csv"), sep=",", index=False)
