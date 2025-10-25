"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
       print("No enough numerical columns for relational plot")
       return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df, color='seagreen', ax=ax)
    ax.set_title(f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}")
    ax.set_xlabel(numeric_cols[0])
    ax.set_ylabel(numeric_cols[1])
    fig.tight_layout()
    fig.savefig('relational_plot.png')
    plt.close(fig)
    return


def plot_categorical_plot(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df[numeric_cols], orient="h", palette="Set2", ax=ax)
    ax.set_title("Box Plot of numeric Features")
    fig.tight_layout()
    fig.savefig("categorical_plot.png")
    plt.close(fig)
    return


def plot_statistical_plot(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
       print("No enough numerical columns for relational plot")
       return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[numeric_cols[0]], kde=True, bins=20, color='skyblue')
    ax.set_title(f"Distribution of {numeric_cols[0]}")
    ax.set_xlabel(numeric_cols[0])
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig('statistical_plot_histogram.png')
    plt.close(fig)
    print("Histogram saved as 'statistical_plot_histogram.png'.")

    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    fig.savefig('statistical_plot_heatmap.png')
    plt.close(fig)
    print("Correlation heatmap saved as 'statistical_plot_heatmap.png'.")
    return

def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print(df.info())
    print(df.describe())
    missing = df.isnull().sum()
    print(missing)
    if missing.any():
       df = df.dropna()
    else:
       print("\n No missing values found")
    print(df.corr(numeric_only=True))
     
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    skew = moments[2]
    kurt = moments[3]
    if skew >2:
       skew_text = "highlt right skewed"
    elif skew < -2:
       skew_text = "highlt left skewed"
    elif skew > 0:
       skew_text = "right skewed"
    elif skew < 0:
       skew_text = "left skewed"
    else:
       skew_text = "not skewed"

    if kurt > 0:
       kurt_text = 'leptokurtic'
    elif kurt < 0:
       kurt_text = 'platykurtic'
    else:
       kurt_text = 'mesokurtic'

    print(f'The data was {skew_text} and {kurt_text}.')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for analysis.")
    col = numeric_cols[0]
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
