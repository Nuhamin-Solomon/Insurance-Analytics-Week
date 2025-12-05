import pandas as pd

def load_data(path):
    """Loads the insurance dataset."""
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print("Error loading data:", e)
