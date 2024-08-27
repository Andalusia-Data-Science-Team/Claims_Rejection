import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.lstm_encoder import LSTMEmbedding

def read_last_date(file_path='src/data_backup/last_updated_creation_date.txt'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
    return None

def append_last_line(new_line,file_path='src/data_backup/last_updated_creation_date.txt'):
    with open(file_path, 'a') as file:
        file.write('\n' + new_line)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def store_dfs(df_visit, df_service,df_diagnose,df_episode,path_date):
    path_date = path_date + '/'
    path_visit = path_date + 'visit.parquet'; path_service =  path_date + 'service.parquet';path_diag =  path_date + 'diag.parquet'; path_episode = path_date + 'episode.parquet'
    df_visit.to_parquet(path_visit);df_service.to_parquet(path_service);df_diagnose.to_parquet(path_diag);df_episode.to_parquet(path_episode)

def load_stored_dfs(path_date):
    path_date = path_date + '/'
    path_visit = path_date + 'visit.parquet'; path_service =  path_date + 'service.parquet';path_diag =  path_date + 'diag.parquet'; path_episode = path_date + 'episode.parquet'
    df_visit = pd.read_parquet(path_visit); df_service = pd.read_parquet(path_service); df_diagnose = pd.read_parquet(path_diag)
    df_episode = pd.read_parquet(path_episode)
    return df_visit, df_service,df_diagnose, df_episode



class DataLoader:
    def __init__(self, source='HJH'):
        self.source = source
        self.PATH = 'data/'+source +'/'

    def load_local(self):

        PATH_1 = self.PATH + 'scanned/Claim_Service.xlsx'
        PATH_2 = self.PATH + 'scanned/Claim_Visit.xlsx'

        df_service = pd.read_excel(PATH_1)
        df_visit   = pd.read_excel(PATH_2)

        return df_service, df_visit

    def _get_transaction_requests(self,df_trans): ## deprecated
        return df_trans[df_trans['TransactionType']=='Request']

    def split_transaction(self,df_trans): ## deprecated
        df_trans_req = df_trans[df_trans['TransactionType']=='Request']
        df_trans_res = df_trans[df_trans['TransactionType']=='Response']
        return df_trans_req, df_trans_res

    def _drop_duplicates(self,df_visit,service_columns):
        for col in list(df_visit.columns):
            if col != 'VISIT_ID' and col in service_columns:
                df_visit.drop(columns = [col],inplace=True)
        return df_visit

    def _add_prefix_to_columns(self, df, prefix):
        df2 = df.copy()
        df2.columns = [prefix + col for col in df2.columns]
        return df2

    def load_data(self,PATH):
        df_service, df_visit = self.load_sourced(PATH)

        df_visit = self._drop_duplicates(df_visit,list(df_service.columns)) ## drop columns duplications
        return df_service, df_visit

    def load_sourced(self,PATH):
        PATH_1 = PATH + '/service.parquet'
        PATH_2 = PATH + '/visit.parquet'

        df_service = pd.read_parquet(PATH_1)
        df_visit   = pd.read_parquet(PATH_2)

        return df_service, df_visit

    def merge_visit_service(self,df_service=None, df_visit=None):
        if df_service is None or df_visit is None: ## reload is required
            df_service, df_visit = self.load_data()
        merged_df = pd.merge(df_service, df_visit, left_on='VISIT_ID',how='left') ## merging on VISIT_ID
        return merged_df


def self_drop_duplicates(df_visit, service_columns):
    for col in list(df_visit.columns):
        if col != 'VISIT_ID' and col in service_columns:
            df_visit.drop(columns=[col], inplace=True)
    return df_visit


def merge_visit_service(df_service=None, df_visit=None):
    merged_df = pd.merge(df_service, df_visit, left_on='VISIT_ID', right_on='VISIT_ID',
                         how='left')  ## merging on VISIT_ID
    return merged_df


def merge_claim(df_visit, df_service, df_episode):
    df_visit_unique = self_drop_duplicates(df_visit, df_service.columns)
    df_claim = merge_visit_service(df_service, df_visit_unique)

    df_episode['VISIT_NO'] = df_episode['Episode_Key'].apply(lambda x: x.split('_')[1])
    df_claim = df_claim.merge(df_episode, how='left', on='VISIT_NO')
    return df_claim


class MergedDataPreprocessing:
    def __init__(self, df):
        self.df = df
        self.lstm_embedding = LSTMEmbedding()

    def _add_list_to_json(self,list_name, values,file_path = 'src/data_backup/label_encoding_items.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        data[list_name] = values
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _read_list_from_json(self,column_name, file_path = 'src/data_backup/label_encoding_items.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get(column_name, None)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {file_path}.")
            return None

    def _truncate_column_values(self, column):
        '''
        :param column: date column value
        :return: utputs only yy/mm
        '''
        df = self.df.copy()
        df[column] = df[column].astype(str).str[:7]

        return df

    def _eliminate_null_claims(self,df_sorted):
        return df_sorted[df_sorted['OUTCOME'].notnull()] ## assert 'APPROVED' or 'PARTIAL'

    def train_test_split(self, id_column='VISIT_ID', test_size=0.2, random_state=None):
        df = self.df
        df = self._eliminate_null_claims(df) ## assert 'Accepted' and 'Rejected' cases

        unique_ids = df[id_column].unique()
        train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

        train_data = df[df[id_column].isin(train_ids)]
        test_data = df[df[id_column].isin(test_ids)]

        return train_data, test_data

    def train_test_split_time(self,id_column = 'CREATION_DATE',test_size= 0.2):
        df = self.df
        df = self._eliminate_null_claims(df) ## assert 'Accepted' and 'Rejected' cases
        df = df.sort_values([id_column])
        train_size = 1 - test_size
        split_time = int(len(df) * train_size)

        train_data, test_data = df[:split_time], df[split_time:]

        return train_data, test_data

    def _label_encode_column(self, column_name, min_count, replace_value='Other'):
        df = self.df.copy()
        counts = df[column_name].value_counts()
        values_to_replace = counts[counts < min_count].index
        df[column_name] = df[column_name].replace(values_to_replace, replace_value)

        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])

        return df[column_name].values


    def _get_parent_family(self, icd10_code):  ##updated with label_items_encoding.json dictionary values (Mid Range)
        icd10_code = str(icd10_code)
        if icd10_code in ['NaN','Nan','None','NULL','nan']:
            return 0
        if icd10_code == 'XX':
            return 1
        if icd10_code[0].upper() == "A" or icd10_code[0].upper() == "B":
            return 2
        elif icd10_code[0].upper() == "C" or (icd10_code[0].upper() == "D" and int(icd10_code[1].upper()) < 5):
            return 3
        elif icd10_code[0].upper() == "D":
            return 4
        elif icd10_code[0].upper() == "E":
            return 5
        elif icd10_code[0].upper() == "F":
            return 6
        elif icd10_code[0].upper() == "G":
            return 7
        elif icd10_code[0].upper() == "H" and int(icd10_code[1].upper()) < 6:
            return 8
        elif icd10_code[0].upper() == "H":
            return 9
        elif icd10_code[0].upper() == "I":
            return 10
        elif icd10_code[0].upper() == "J":
            return 11
        elif icd10_code[0].upper() == "K":
            return 12
        elif icd10_code[0].upper() == "L":
            return 13
        elif icd10_code[0].upper() == "M":
            return 14
        elif icd10_code[0].upper() == "N":
            return 15
        elif icd10_code[0].upper() == "O":
            return 16
        elif icd10_code[0].upper() == "P":
            return 17
        elif icd10_code[0].upper() == "Q":
            return 18
        elif icd10_code[0].upper() == "R":
            return 19
        elif icd10_code[0].upper() == "S" or icd10_code[0].upper() == "T":
            return 20
        elif icd10_code[0].upper() == "V" or icd10_code[0].upper() == "Y":
            return 21
        elif icd10_code[0].upper() == "Z":
            return 22
        else:
            return 23

    def _categorize_age(self,age):
        if age <= 2:
            return 'Kids (0-2)'
        elif 3 <= age <= 10:
            return 'Age 3-10'
        elif 11 <= age <= 17:
            return 'Age 11-17'
        elif 18 <= age <= 24:
            return 'Age 18-24'
        elif 25 <= age <= 34:
            return 'Age 25-34'
        elif 35 <= age <= 44:
            return 'Age 35-44'
        elif 45 <= age <= 54:
            return 'Age 45-54'
        elif 55 <= age <= 64:
            return 'Age 55-64'
        else:
            return 'Age 65+'

    def _preprocess_service(self,eng_sentence):
        eng_sentence = eng_sentence.split('-')[0]
        return eng_sentence

    def _replace_strings_in_column(self, undefined_inp:str,replacement_value = 0):
        if type(undefined_inp) == str:
            undefined_inp = replacement_value
        return undefined_inp

    def columns_prep(self,service_encoding=False,icd_encoding=False):
        df = self.df
        LIST_ENCODED_COLS = ["PATIENT_GENDER","ICD10","EMERGENCY_INDICATOR","PATIENT_NATIONALITY","PATIENT_MARITAL_STATUS","CLAIM_TYPE","NEW_BORN","TREATMENT_TYPE"]
        for column in LIST_ENCODED_COLS:
            column_encoding = self._read_list_from_json(column_name=column)
            df[column]= df[column].replace(column_encoding)

            if column != 'ICD10': ## Only ICD10 can have string in those columns
                df[column] = df[column].apply(self._replace_strings_in_column)

        if 'PatientAgeRange' not in df.columns:   ## check not to repeat preprocessing
            df['PatientAgeRange'] = df['PATIENT_AGE'].astype(int).apply(self._categorize_age)
            age_encoding = self._read_list_from_json(column_name='AGE_RANGE')
            df['PatientAgeRange']= df['PatientAgeRange'].replace(age_encoding)

        df['PROVIDER_DEPARTMENT'] = df.PROVIDER_DEPARTMENT.apply(self._preprocess_service)
        df['DURATION'] = df['DURATION'].fillna(0)

        ### not recommended
        if service_encoding:
            df.SERVICE_DESCRIPTION = self._label_encode_column(column_name='SERVICE_DESCRIPTION', min_count=15)
        if icd_encoding:
            # temp comment this line as the data sent was already encoded
            if type(df.ICD10.iloc[0]) == str:
                df['ICD10']  = df['ICD10'].apply(lambda x:self._get_parent_family(x))
        self.df = df
        return self.df

    def column_embedding(self, df1, is_service=True,service_columns = ['SERVICE_DESCRIPTION', 'SERVICE_TYPE','UNIT','UNIT_TYPE','TIMES','PER', 'OASIS_IOS_DESCRIPTION', 'PROVIDER_DEPARTMENT']):
        if is_service:
            textual_col = service_columns
            df1['CombinedText'] = df1[textual_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            arr2 = self.lstm_embedding.embedding_vector(df1['CombinedText'].tolist(), reload_model=True)
            new_cols_names = ['CombinedText' + str(i + 1) for i in range(arr2.shape[1])]
            df2 = pd.DataFrame(arr2)
            df2.columns = new_cols_names
            for col in df2.columns:
                df1.loc[:, col] = df2[col].values
            to_drop = textual_col + ['CombinedText']
            df1.drop(columns=to_drop, inplace=True)

        else:
            textual_col = 'ICD10'
            arr2 = self.lstm_embedding.embedding_vector(df1[textual_col].tolist(), reload_model=True)
            new_cols_names = ['ICDText' + str(i + 1) for i in range(arr2.shape[1])]
            df2 = pd.DataFrame(arr2)
            df2.columns = new_cols_names
            for col in df2.columns:
                df1.loc[:, col] = df2[col].values
            to_drop = [textual_col]
            df1.drop(columns=to_drop, inplace=True)

        return df1
    def store_current_columns(self,df_index,encoding_values:dict):
        self._add_list_to_json(list_name=df_index,values=encoding_values)