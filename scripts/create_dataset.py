import pandas as pd

from dam_segmentation.feature_extraction import create_dataset

# 1 - Create binary dataset
# Create training dataset
# train_images = "data/train/images/"
# train_labels = "data/train/labels_binary/"
# train_output = "data/train/binary_training.parquet"
# dataset = create_dataset(train_images, train_labels)
# dataset.to_parquet(train_output)


# # Create test dataset
# test_images = "data/test/images/"
# test_labels = "data/test/labels_binary/"
# test_output = "data/test/binary_test.parquet"
# dataset = create_dataset(test_images, test_labels)
# dataset.to_parquet(test_output)

# 2 - Create multiclass dataset
# Create training dataset
train_images = "data/train/images/"
train_labels = "data/train/labels_multiclass/"
train_output = "data/train/multiclass_training.parquet"
dataset = create_dataset(train_images, train_labels)
dataset.to_parquet(train_output)


# Create test dataset
test_images = "data/test/images/"
test_labels = "data/test/labels_multiclass/"
test_output = "data/test/multiclass_test.parquet"
dataset = create_dataset(test_images, test_labels)
dataset.to_parquet(test_output)
