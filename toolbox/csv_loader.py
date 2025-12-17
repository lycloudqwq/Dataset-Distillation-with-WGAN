import pandas as pd


def load_data(csvfile):
    df = pd.read_csv(
        csvfile,
        names=['x', 'y', 'bits'],
        header=None,
    )

    return {
        "x": df["x"].to_numpy(),
        "y": df["y"].to_numpy(),
        "bits": df["bits"].to_numpy(dtype=int)
    }
