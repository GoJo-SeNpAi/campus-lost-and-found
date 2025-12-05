# src/utils.py
import pandas as pd
import os

def load_dataset(csv_path="data/dataset.csv"):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["id","kind","description","image_path","object","color"])
    return pd.read_csv(csv_path)
