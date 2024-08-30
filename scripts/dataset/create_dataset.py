import pandas as pd

from damseg_ml.feature_extraction import create_dataset

# 1 - Create binary dataset
# Create training dataset
train_images = "data/train/images/"
train_labels = "data/train/labels_binary/"
train_output = "data/train/binary_training.csv"
dataset = create_dataset(train_images, train_labels)
dataset.to_csv(train_output, index=False)


# Create test dataset
test_images = "data/test/images/"
test_labels = "data/test/labels_binary/"
test_output = "data/test/binary_test.csv"
dataset = create_dataset(test_images, test_labels)
dataset.to_csv(test_output, index=False)
dataset.to_csv(test_output, index=False)
