from dam_segmentation.feature_extraction import create_dataset

# ----------------------- Create Binary Tabular Dataset ---------------------- #

train_images = "../data/dam-segmentation/train/images/"
train_labels = "../data/dam-segmentation/train/mask_binary"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "../data/dam-segmentation/train_data_binary.parquet",
    index=False,
)

val_images = "../data/dam-segmentation/val/images/"
val_labels = "../data/dam-segmentation/val/mask_binary/"

val = create_dataset(val_images, val_labels)

val.to_parquet(
    "../data/dam-segmentation/val_data_binary.parquet",
    index=False,
)

test_images = "../data/dam-segmentation/test/images/"
test_labels = "../data/dam-segmentation/test/mask_binary"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "../data/dam-segmentation/test_data_binary.parquet",
    index=False,
)

# -------------------- Create Multi-Class Tabular Dataset -------------------- #

train_images = "../data/dam-segmentation/train/images/"
train_labels = "../data/dam-segmentation/train/mask_multiclass"

train = create_dataset(train_images, train_labels)

train.to_parquet(
    "../data/dam-segmentation/train_data_multiclass.parquet",
    index=False,
)

val_images = "../data/dam-segmentation/val/images/"
val_labels = "../data/dam-segmentation/val/mask_multiclass/"

val = create_dataset(val_images, val_labels)

val.to_parquet(
    "../data/dam-segmentation/val_data_multiclass.parquet",
    index=False,
)

test_images = "../data/dam-segmentation/test/images/"
test_labels = "../data/dam-segmentation/test/mask_multiclass"

test = create_dataset(test_images, test_labels)

test.to_parquet(
    "../data/dam-segmentation/test_data_multiclass.parquet",
    index=False,
)
