import pandas as pd

# Step 1: Load your dataset
df = pd.read_parquet("data/train/multiclass_training.parquet")

# Step 2: Identify the label with the fewest samples (underrepresented)
label_counts = df["label"].value_counts()
min_label_count = label_counts.min()

# Step 3: Downsample the overrepresented label
downsampled_df_list = []
for label_value in label_counts.index:
    label_subset = df[df["label"] == label_value]
    downsampled_label_subset = label_subset.sample(
        n=min_label_count, random_state=42
    )
    downsampled_df_list.append(downsampled_label_subset)

# Step 4: Combine the downsampled data
balanced_df = pd.concat(downsampled_df_list)

# Optionally shuffle the dataset to mix the labeles
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
    drop=True
)

# Step 5: Save the balanced dataset as a new CSV file
balanced_df.to_csv("balanced_dataset.csv", index=False)

# Optionally print the label distribution to verify
print(balanced_df["label"].value_counts())
balanced_df.to_parquet("data/train/multiclass_balanced_training.parquet")
