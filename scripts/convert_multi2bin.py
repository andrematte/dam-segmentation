# This scripts is used to convert the multiclass dataset labels to binary labels
# The multiclass dataset has 4 classes: 0, 1, 2, 3
# The binary dataset has 2 classes: 0, 1
# The binary dataset is created by changing the labels of all rows where label in [2, 3] to 0

import pandas as pd

train = "../data/train_data_multiclass_downsampled.parquet"
val = "../data/val_data_multiclass.parquet"
test = "../data/test_data_multiclass.parquet"

train = pd.read_parquet(train)
val = pd.read_parquet(val)
test = pd.read_parquet(test)

# Change the labels of all rows where label in [2, 3] to 0
train.loc[train["label"].isin([2, 3]), "label"] = 0
val.loc[val["label"].isin([2, 3]), "label"] = 0
test.loc[test["label"].isin([2, 3]), "label"] = 0

train.to_parquet("../data/train_data_binary_downsampled.parquet", index=False)
val.to_parquet("../data/val_data_binary.parquet", index=False)
test.to_parquet("../data/test_data_binary.parquet", index=False)
