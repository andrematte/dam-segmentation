from dam_segmentation.feature_extraction import create_dataset

# ----------------------- Create Binary Tabular Dataset ---------------------- #

train_images = "/Users/andrematte/Data/nesa-dataset/train/images/"
train_labels = "/Users/andrematte/Data/nesa-dataset/train/mask_binary"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "../data/train_data_binary.parquet",
    index=False,
)

val_images = "/Users/andrematte/Data/nesa-dataset/val/images/"
val_labels = "/Users/andrematte/Data/nesa-dataset/val/mask_binary/"

val = create_dataset(val_images, val_labels)

val.to_parquet(
    "../data/val_data_binary.parquet",
    index=False,
)

test_images = "/Users/andrematte/Data/nesa-dataset/test/images/"
test_labels = "/Users/andrematte/Data/nesa-dataset/test/mask_binary"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "../data/test_data_binary.parquet",
    index=False,
)

# -------------------- Create Multi-Class Tabular Dataset -------------------- #

train_images = "/Users/andrematte/Data/nesa-dataset/train/images/"
train_labels = "/Users/andrematte/Data/nesa-dataset/train/mask_multiclass"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "../data/train_data_multiclass.parquet",
    index=False,
)

val_images = "/Users/andrematte/Data/nesa-dataset/val/images/"
val_labels = "/Users/andrematte/Data/nesa-dataset/val/mask_multiclass/"

val = create_dataset(val_images, val_labels)

val.to_parquet(
    "../data/val_data_multiclass.parquet",
    index=False,
)

test_images = "/Users/andrematte/Data/nesa-dataset/test/images/"
test_labels = "/Users/andrematte/Data/nesa-dataset/test/mask_multiclass"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "../data/test_data_multiclass.parquet",
    index=False,
)
