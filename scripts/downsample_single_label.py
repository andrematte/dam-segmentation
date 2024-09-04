import pandas as pd

# Step 1: Load your dataset
df = pd.read_parquet(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/multiclass_training.parquet"
)

# Step 2: Identify the label with the fewest samples (underrepresented)
label_counts = df["label"].value_counts()
min_label_name = 0
min_label_count = label_counts[min_label_name]

# Step 3: Downsample the overrepresented label
downsampled_df_list = []
for label_value in label_counts.index:
    label_subset = df[df["label"] == label_value]
    if label_counts[label_value] > min_label_count:
        downsampled_label_subset = label_subset.sample(
            n=min_label_count, random_state=42
        )
        downsampled_df_list.append(downsampled_label_subset)
    else:
        downsampled_df_list.append(label_subset)

# Step 4: Combine the downsampled data
balanced_df = pd.concat(downsampled_df_list)

# Optionally shuffle the dataset to mix the labeles
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
    drop=True
)

print(balanced_df["label"].value_counts())
balanced_df.to_parquet(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/multiclass_balanced_training.parquet"
)
