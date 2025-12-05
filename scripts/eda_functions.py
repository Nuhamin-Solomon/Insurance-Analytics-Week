import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOTS_DIR = "plots"

def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def plot_hist(df, column):
    ensure_plots_dir()
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Distribution")
    plt.savefig(f"{PLOTS_DIR}/{column}_hist.png")
    plt.close()

def plot_bar(df, column):
    ensure_plots_dir()
    plt.figure(figsize=(6,4))
    df[column].value_counts().plot(kind="bar")
    plt.title(f"{column} Count")
    plt.savefig(f"{PLOTS_DIR}/{column}_bar.png")
    plt.close()

def plot_scatter(df, x, y):
    ensure_plots_dir()
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{y} vs {x}")
    plt.savefig(f"{PLOTS_DIR}/{y}_vs_{x}.png")
    plt.close()
