import pandas as pd
from src.data_local import DataLoader
from src.model_train import encode_label

train_columns = []

def get_cashed_input():
    return pd.read_parquet('data/HJH/prq/df.parquet')


def get_input():
    data_loader = DataLoader()
    df_visit_service = data_loader.merge_visit_service() ## High variance filter is done already
    return df_visit_service

def get_train_test_split():
    df_train = pd.read_parquet('data/HJH/prq/train.parquet')
    df_test  = pd.read_parquet('data/HJH/prq/test.parquet')
    return df_train, df_test


def get_training_inputs(df_train, df_test):
    labels_cols = ['OUTCOME','SUBMIT_CLAIM_MESSAGE']
    X_train = df_train.drop(columns=labels_cols);  y_train = df_train[labels_cols]
    X_test = df_test.drop(columns=labels_cols);   y_test = df_test[labels_cols]

    y_test['STATUS'] = encode_label(y_test['STATUS'].tolist());   y_train['STATUS'] = encode_label(y_train['STATUS'].tolist())

    return X_train, y_train, X_test, y_test
