import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_parquet(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/binary_reduced_training.parquet"
).drop(columns=["label"])

# selected = data[data.columns[:9]] # Spectral
# selected = data[data.columns[10:58]]  # Gabor
# selected = data[data.columns[58:69]]  # Filters
selected = data[data.columns[69:]]  # GLCM


correlation = selected.corr()

selected_labels = correlation.columns
selected_data = correlation.to_numpy()

annot = selected_data.round(2)
# annot = np.array(
#     [["{:.1f}%".format(value) for value in row] for row in percent_data]
# )

plt.figure(figsize=(8, 6))
sns.heatmap(
    selected_data,
    annot=annot,
    fmt="",
    cmap="coolwarm",
    xticklabels=[f"[{str(i)}]" for i in range(len(selected_labels))],
    yticklabels=[
        f"{selected_labels[i]} [{str(i)}]" for i in range(len(selected_labels))
    ],
    cbar=True,
    linewidths=3,
    linecolor="white",
    vmin=-1,
    vmax=1,
)

# plt.xlabel("Predicted label")
# plt.ylabel("True label")

plt.show()
