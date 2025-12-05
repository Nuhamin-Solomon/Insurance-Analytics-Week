import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(df, column):
    plt.figure(figsize=(8,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_bar(df, column):
    plt.figure(figsize=(8,4))
    df[column].value_counts().plot(kind='bar')
    plt.title(f"Count of {column}")
    plt.show()
