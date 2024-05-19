import pandas as pd

from src.data_application import DataFrameProcessor, model_columns
from src.data_local import DataLoader
from src.data_application import train_columns
from src.model_train import encode_label
def get_input():
    data_loader = DataLoader()
    df_trans_item = data_loader.merge_item_trans()
    processor = DataFrameProcessor(df_trans_item)
    df_trans_item = processor.filter_high_variance_features()
    return df_trans_item[model_columns]

def get_train_test_split():
    df_train = pd.read_excel('data/SplittedData/SNB_train_data.xlsx')
    df_test  = pd.read_excel('data/SplittedData/SNB_test_data.xlsx')
    return df_train, df_test


def get_training_inputs(df_train, df_test):
    X_train = df_train[train_columns[:-1]];    y_train = df_train[train_columns[-1]]
    X_test = df_test[train_columns[:-1]];    y_test = df_test[train_columns[-1]]

    y_test = encode_label(y_test);     y_train = encode_label(y_train)

    return X_train, y_train, X_test, y_test
