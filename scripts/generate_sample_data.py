import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Province": np.random.choice(["ON", "BC", "QC", "AB", "MB"], n),
        "VehicleType": np.random.choice(["Sedan", "SUV", "Truck", "Van"], n),
        "TotalPremium": np.random.uniform(300, 2000, n).round(2),
        "TotalClaims": np.random.uniform(0, 10000, n).round(2)
    })

    df["LossRatio"] = (df["TotalClaims"] / df["TotalPremium"]).round(2)

    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/insurance_data.csv", index=False)
    print("Dataset created!")
