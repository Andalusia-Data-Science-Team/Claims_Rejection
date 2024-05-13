import pandas as pd

from src.data_application import DataFrameProcessor, model_columns
from src.data_local import DataLoader

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