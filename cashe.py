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
    if 'BART_LABEL' in df_test.columns:
        labels_cols.append('BART_LABEL')

    X_train = df_train.drop(columns=labels_cols);  y_train = df_train[labels_cols]
    X_test = df_test.drop(columns=labels_cols);   y_test = df_test[labels_cols]

    y_train.loc[:, 'OUTCOME'] = encode_label(y_train['OUTCOME'].tolist()); y_test.loc[:, 'OUTCOME'] = encode_label(y_test['OUTCOME'].tolist())
    return X_train, y_train, X_test, y_test


def drop_nomodel_columns(df):
    ## list of columns not needed in the modeling.
    list_handler = ['PERIOD', 'DATE', '_ID', '_NO', '_API', 'ATTACH', 'PATIENT_DOB', '_USER', 'TREATMENT_SUB_TYPE',
                    'ELIGIBILITY_RESPONSE_SYSTEM', 'DOCTOR_LICENSE', 'OUTCOME', '_QUANTITY', 'RES_STATUS', 'NOTES',
                    '_LICENSE','APPROVED_QUNATITY','_Key']

    df2 = df.copy()
    for substring in list_handler:
        columns_to_drop = [col for col in df2.columns if substring in col]
        df2 = df2.drop(columns=columns_to_drop)

    df2 = df2.drop(columns=['STATUS'])
    return df2