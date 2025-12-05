import pandas as pd
from eda_functions import plot_hist, plot_bar, plot_scatter

df = pd.read_csv("data/insurance_data.csv")

# Histograms
plot_hist(df, "Age")
plot_hist(df, "TotalPremium")
plot_hist(df, "TotalClaims")
plot_hist(df, "LossRatio")

# Bar charts
plot_bar(df, "Gender")
plot_bar(df, "Province")
plot_bar(df, "VehicleType")

# Scatter plot
plot_scatter(df, "TotalPremium", "TotalClaims")
