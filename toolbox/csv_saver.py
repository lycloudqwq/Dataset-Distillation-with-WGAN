import numpy as np
import pandas as pd


def save_data(x, y, bits, csvfile="distilled_dataset.csv"):
    x = np.asarray(x)
    y = np.asarray(y)
    bits = np.asarray(bits)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "bits": bits.astype(int),
    })

    df.to_csv(csvfile, index=False, header=False)
