from dam_segmentation.feature_extraction import create_dataset

# ----------------------- Create Binary Tabular Dataset ---------------------- #

train_images = "/Users/andrematte/Data/nesa-dataset/images/train"
train_labels = "/Users/andrematte/Data/nesa-dataset/mask_binary/train"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "./data/train_data_binary.parquet",
    index=False,
)

test_images = "/Users/andrematte/Data/nesa-dataset/images/test"
test_labels = "/Users/andrematte/Data/nesa-dataset/mask_binary/test"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "./data/test_data_binary.parquet",
    index=False,
)

# -------------------- Create Multi-Class Tabular Dataset -------------------- #

train_images = "/Users/andrematte/Data/nesa-dataset/images/train"
train_labels = "/Users/andrematte/Data/nesa-dataset/mask_multiclass/train"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "./data/train_data_multiclass.parquet",
    index=False,
)

test_images = "/Users/andrematte/Data/nesa-dataset/images/test"
test_labels = "/Users/andrematte/Data/nesa-dataset/mask_multiclass/test"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "./data/test_data_multiclass.parquet",
    index=False,
)
