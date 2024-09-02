import pandas as pd
from src.data_local import DataLoader
from src.model_train import encode_label
train_columns = []

def drop_duplicated_claims(df): ## reduces 20% of data
    df = df.sort_values( by =['CREATION_DATE'])
    cols = ['VISIT_ID','NET_WITH_VAT', 'SERVICE_DESCRIPTION', 'QTY', 'LINE_CLAIMED_AMOUNT_SAR', 'LINE_ITEM_DISCOUNT', 'NET_VAT_AMOUNT', 'PATIENT_VAT_AMOUNT', 'VAT_PERCENTAGE', 'TREATMENT_TYPE_INDICATOR', 'SERVICE_TYPE', 'DURATION', 'QTY_STOCKED_UOM', 'OASIS_IOS_DESCRIPTION', 'UNIT_PRICE_STOCKED_UOM', 'UNIT_PRICE_NET', 'DISCOUNT_PERCENTAGE', 'EMERGENCY_INDICATOR', 'PROVIDER_DEPARTMENT_CODE', 'PROVIDER_DEPARTMENT', 'DOCTOR_SPECIALTY_CODE', 'DOCTOR_CODE', 'PATIENT_AGE', 'UNIT_OF_AGE', 'PATIENT_NATIONALITY', 'PATIENT_MARITAL_STATUS', 'PATIENT_GENDER', 'CLAIM_TYPE', 'TOTAL_CLAIMED_AMOUNT_SAR', 'TOTAL_DISCOUNT', 'TOTAL_DEDUCTIBLE', 'TOTAL_PATIENT_VATAMOUNT', 'DEPARTMENT_TYPE', 'TREATMENT_TYPE', 'PURCHASER_CODE', 'NEW_BORN', 'ICD10']
    df.drop_duplicates(cols,keep='last',inplace=True)
    return df

def read_cashed_original():
    df = pd.read_parquet('data/HJH/prq/df.parquet')
    return df

def get_cashed_input():
    #df = pd.read_parquet('data/HJH/prq/df.parquet')
    df = pd.read_parquet('data/HJH/13-06-2024/df.parquet')
    df = drop_duplicated_claims(df)
    df = df[df['OUTCOME'].isin(['APPROVED','REJECTED','PARTIAL'])]
    return df

def get_input():
    data_loader = DataLoader()
    df_visit_service = data_loader.merge_visit_service() ## High variance filter is done already
    return df_visit_service

def get_train_test_split(path):
    train_p = path + '/train.parquet'; test_p = path + '/test.parquet'
    df_train = pd.read_parquet(train_p)
    df_train = drop_duplicated_claims(df_train)
    df_test  = pd.read_parquet(test_p)
    df_test  = drop_duplicated_claims(df_test)

    return df_train, df_test

#def check_test_data():

def get_training_inputs(df_train, df_test):
    labels_cols = ['OUTCOME','SUBMIT_CLAIM_MESSAGE']
    if 'NPHIES_LABEL' in df_test.columns:
        labels_cols.append('NPHIES_LABEL')

    X_train = df_train.drop(columns=labels_cols);  y_train = df_train[labels_cols]
    X_test = df_test.drop(columns=labels_cols);   y_test = df_test[labels_cols]

    y_train.loc[:, 'OUTCOME'] = encode_label(y_train['OUTCOME'].tolist()); y_test.loc[:, 'OUTCOME'] = encode_label(y_test['OUTCOME'].tolist())
    return X_train, y_train, X_test, y_test

def get_testing_inputs(df_test):
    labels_cols = ['OUTCOME','SUBMIT_CLAIM_MESSAGE']
    if 'NPHIES_LABEL' in df_test.columns:
        labels_cols.append('NPHIES_LABEL')

    X_test = df_test.drop(columns=labels_cols);   y_test = df_test[labels_cols]

    y_test.loc[:, 'OUTCOME'] = encode_label(y_test['OUTCOME'].tolist())
    return X_test, y_test

def drop_nomodel_columns(df):
    ## list of columns not needed in the modeling.
    list_handler = ['PERIOD', 'DATE', '_ID', '_NO', '_API', 'ATTACH', 'PATIENT_DOB', '_USER', 'TREATMENT_SUB_TYPE',
                    'ELIGIBILITY_RESPONSE_SYSTEM', 'DOCTOR_LICENSE', 'OUTCOME', '_QUANTITY', 'RES_STATUS', 'NOTES',
                    '_LICENSE','APPROVED_QUNATITY','_Key']

    df2 = df.copy()
    for substring in list_handler:
        columns_to_drop = [col for col in df2.columns if substring in col]
        columns_to_drop = [col for col in columns_to_drop if col not in ['CONTRACT_NO','CO_INSURANCE']]
        df2 = df2.drop(columns=columns_to_drop)

    cols_drop = ['HIS_INSURANCE_CODE',  'TOTAL_NET_AMOUNT', 'TOTAL_NET_VAT_AMOUNT', 'TOTAL_CLAIMED_AMOUNT', 'LINE_CLAIMED_AMOUNT',  'NET_AMOUNT','SERVICE_CATEGORY', 'UNIT_PRICE','STATUS', 'CO_PAY']
    for col in cols_drop:
        df2 = df2.drop(columns=[col])

    return df2