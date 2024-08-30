import pandas as pd

from damseg_ml.feature_extraction import create_dataset

# 1 - Create binary dataset
# Create sample dataset
train_images = "data/sample/images/"
train_labels = "data/sample/labels_binary/"
dataset = create_dataset(train_images, train_labels)
dataset.to_csv("data/sample/dataset_binary.csv", index=False)
dataset.to_csv("data/sample/dataset_binary.csv", index=False)
