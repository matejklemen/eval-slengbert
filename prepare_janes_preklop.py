import os
import re
import datasets

# https://gist.github.com/gruber/8891611
import numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold

URL_REGEX = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"


def preprocess_text(s):
    _s = re.sub(r"@(\w|\d)+", "[MENTION]", s)  # mentions
    _s = re.sub(URL_REGEX, "[URL]", _s)  # urls
    _s = re.sub("(“|‘|'|«|»|„|”)", "\"", _s)  # quotations
    _s = _s.strip()

    return _s


def map_labels(labels):
    mapped_labels = []
    for _lbl in labels:
        if _lbl.startswith(("B-", "I-")):
            prefix = _lbl[:2]
            mapped_labels.append(f"{prefix}code_switched")
        else:
            mapped_labels.append("O")

    return mapped_labels


if __name__ == "__main__":
    SCHEME = "binary_token"
    assert SCHEME in ["binary_token", "multiclass_token"]

    data = datasets.load_dataset("cjvt/janes_preklop", split="train")
    data = data.to_pandas()

    # Hack: some oddly encoded emoji is causing havoc in the modeling code - replace it with "..."
    data["words"] = data["words"].apply(lambda _words:
                                        list(map(lambda _w: _w if _w != "\U000fe347\U000fe347\U000fe347" else "...", _words)))

    data["words"] = data["words"].apply(lambda _words: list(map(preprocess_text, _words)))
    if SCHEME == "multiclass_token":
        # map default -> O
        data["mapped_language"] = data["language"].apply(lambda _tags:
                                                         list(map(lambda _tag: _tag if _tag != "default" else "O", _tags)))
    else:
        data["mapped_language"] = data["language"].apply(map_labels)
    data = Dataset.from_pandas(data)

    TARGET_DATA_DIR = f"janes_preklop-preprocessed-{SCHEME}"

    RAND_SEED = 17
    np.random.seed(RAND_SEED)
    kfold = KFold(n_splits=10, shuffle=True, random_state=RAND_SEED)
    os.makedirs(TARGET_DATA_DIR, exist_ok=True)

    count = np.zeros(len(data))
    for idx_fold, (train_indices, test_indices) in enumerate(kfold.split(data["words"])):
        dev_indices = train_indices[-len(test_indices):]
        print(f"Fold #{idx_fold}:\n"
              f"\t{len(train_indices)}\n"
              f"\t{len(dev_indices)}\n"
              f"\t{len(test_indices)}")

        FOLD_DATA_DIR = os.path.join(TARGET_DATA_DIR, f"fold{idx_fold}")
        os.makedirs(FOLD_DATA_DIR, exist_ok=True)

        data.select(train_indices).to_json(os.path.join(FOLD_DATA_DIR, "train.jsonl"), lines=True, orient="records",
                                           force_ascii=False)
        data.select(dev_indices).to_json(os.path.join(FOLD_DATA_DIR, "dev.jsonl"), lines=True, orient="records",
                                         force_ascii=False)
        data.select(test_indices).to_json(os.path.join(FOLD_DATA_DIR, "test.jsonl"), lines=True, orient="records",
                                          force_ascii=False)


