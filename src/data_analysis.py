import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

class DataUnivariateVisualization:
    def __init__(self, df):
        self.df = df

    def filter_column_by_value_count(self, column_name, threshold):
        value_counts = self.df[column_name].value_counts()
        to_keep = value_counts[value_counts >= threshold].index
        filtered_df = self.df[self.df[column_name].isin(to_keep)]
        return filtered_df

    def plot_value_counts(self, column_name, xlim=None):
        value_counts = self.df[column_name].value_counts()
        if xlim:
            value_counts = value_counts[value_counts.index <= xlim]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
        plt.xlabel(column_name)
        plt.ylabel('Counts')
        plt.title(f'Value Counts of {column_name}')
        plt.xticks(rotation=45)
        plt.show()

    def plot_distribution(self, column_name):
        column_counts = self.df.groupby(column_name).size()

        plt.figure(figsize=(10, 6))
        sns.histplot(column_counts, bins=range(1, column_counts.max() + 1), kde=False, color='blue')
        plt.xscale('log')
        plt.xlabel(f'Number of Rows per {column_name}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Rows per {column_name}')

        specific_ticks = [1, 5, 10, 15, 20] + [10**i for i in range(1, int(np.log10(column_counts.max())) + 1)]
        specific_ticks = sorted(set(specific_ticks))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self.log_format))

        plt.xticks(specific_ticks)
        plt.show()

    @staticmethod
    def log_format(x, pos):
        return f'{int(x)}' if x >= 1 else f'{x:.1f}'

    def plot_boxplot(self, column_name, threshold=None):
        plt.figure(figsize=(10, 6))
        if threshold:
            df2 = self.filter_column_by_value_count(column_name, threshold)
        else:
            df2 = self.df.copy()

        sns.boxplot(data=df2, y=column_name)
        plt.xlabel('Service')
        plt.ylabel('Net Value')
        plt.title(f'Box Plot of Net Value for {column_name}')
        plt.show()

    def plot_histogram(self, column_name, threshold=None):
        plt.figure(figsize=(10, 6))
        if threshold:
            df2 = self.filter_column_by_value_count(column_name, threshold)
        else:
            df2 = self.df.copy()

        sns.histplot(data=df2, x=column_name, kde=True)
        plt.xlabel(f'{column_name}')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Net Value for {column_name}')
        plt.show()

    def plot_violin_distribution(self, column_name, threshold=None):
        plt.figure(figsize=(14, 7))

        if threshold:
            df2 = self.filter_column_by_value_count(column_name, threshold)
        else:
            df2 = self.df.copy()

        sns.violinplot(x=df2[column_name], color='y')
        plt.xlabel(f'{column_name}')
        plt.ylabel('Density')
        plt.title(f'Violin Plot of {column_name}')
        plt.tight_layout()
        plt.show()

    def analyze_null_values(self, column_name):
        total_nulls = self.df[column_name].isnull().sum()
        total_rows = len(self.df)
        null_percentage = (total_nulls / total_rows) * 100

        print(f"Column: {column_name}")
        print(f"Total Null Values: {total_nulls}")
        print(f"Percentage of Null Values: {null_percentage:.2f}%")

