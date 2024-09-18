from sklearn.model_selection import train_test_split

from dam_segmentation.feature_extraction import create_dataset

# ----------------------- Create Binary Tabular Dataset ---------------------- #

images = "./data/images"
labels = "./data/labels_binary"

dataset = create_dataset(images, labels)

train, test = train_test_split(
    dataset, test_size=0.2, random_state=42, stratify="label"
)

train.to_parquet(
    ".data/train_data_binary.parquet",
    index=False,
)
test.to_parquet(
    ".data/test_data_binary.parquet",
    index=False,
)

# -------------------- Create Multi-Class Tabular Dataset -------------------- #

labels = "./data/labels_multiclass"

dataset = create_dataset(images, labels)

train, test = train_test_split(
    dataset, test_size=0.2, random_state=42, stratify="label"
)

train.to_parquet(
    ".data/train_data_multiclass.parquet",
    index=False,
)
test.to_parquet(
    ".data/test_data_multiclass.parquet",
    index=False,
)
