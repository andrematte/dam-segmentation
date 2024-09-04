import pandas as pd
from sklearn.model_selection import train_test_split

from dam_segmentation.feature_extraction import create_dataset

# Create full dataset
images = "/Users/andrematte/Data/nesa-dataset/images"
labels = "/Users/andrematte/Data/nesa-dataset/mask_multiclass"

dataset = create_dataset(images, labels)

train, test = train_test_split(
    dataset, test_size=0.2, random_state=42, stratify="label"
)

train.to_parquet(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/train_data.parquet",
    index=False,
)
test.to_parquet(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/test/test_data.parquet",
    index=False,
)
