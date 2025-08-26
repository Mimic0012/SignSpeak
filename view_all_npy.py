# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your data folder
DATA_DIR = Path(r"C:\Users\avinash\source\repos\Signlanguage\data")

# Loop over all .npy files
for npy_file in DATA_DIR.glob("*.npy"):
    print(f"\n=== {npy_file.name} ===")
    data = np.load(npy_file)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    # If 21 hand landmarks × 3 coords flattened
    if data.shape[1] == 63:
        sample = data[0].reshape(21, 3)
        plt.figure(figsize=(4, 4))
        plt.scatter(sample[:, 0], sample[:, 1], c='red')
        for i, (x, y, z) in enumerate(sample):
            plt.text(x, y, str(i), fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(f"{npy_file.name} - Sample 0")
        plt.show()
    else:
        print("Preview not supported for shape:", data.shape)