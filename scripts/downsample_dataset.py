import pandas as pd
from imblearn.over_sampling import ADASYN

# Load the train datasets
train = pd.read_parquet("../data/dam-segmentation/train_data_multiclass.parquet")
train["label"].value_counts()

# Downsample the majority classes and upsample the minority classes
SAMPLE_TARGET_VALUE = 707843

# Randomly sample classes where label==0 or label==1
train_0 = train[train["label"] == 0].sample(SAMPLE_TARGET_VALUE)
train_1 = train[train["label"] == 1].sample(SAMPLE_TARGET_VALUE)
train_2 = train[train["label"] == 2]
train_3 = train[train["label"] == 3]

train = pd.concat([train_0, train_1, train_2, train_3])

X = train.drop(columns=["label"])
y = train["label"]

# Oversample the minotiry class with ADASYN
oversampler = ADASYN()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

train = pd.concat([X_resampled, y_resampled], axis=1)

# Shuffle rows
train = train.sample(frac=1).reset_index(drop=True)

print(train["label"].value_counts())

# Save to parquet
train.to_parquet(
    "../data/dam-segmentation/train_data_multiclass_downsampled.parquet",
    index=False,
)
