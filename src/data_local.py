import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.lstm_encoder import LSTMEmbedding


class DataLoader:
    def __init__(self, source='SNB'):
        self.source = source
        self.PATH = 'data/'+source +'/'

    def load_local(self):
        PATH_err  = self.PATH + 'ClaimResponseErrorsLog.xlsx'
        PATH_item = self.PATH + 'ClaimItem.xlsx'
        PATH_trans = self.PATH + 'ClaimTransaction.xlsx'

        df_error = pd.read_excel(PATH_err)
        df_item = pd.read_excel(PATH_item)
        df_trans = pd.read_excel(PATH_trans)

        return df_trans, df_item, df_error

    def _get_transaction_requests(self,df_trans):
        return df_trans[df_trans['TransactionType']=='Request']


    def split_transaction(self,df_trans):
        df_trans_req = df_trans[df_trans['TransactionType']=='Request']
        df_trans_res = df_trans[df_trans['TransactionType']=='Response']
        return df_trans_req, df_trans_res

    def _add_prefix_to_columns(self, df, prefix):
        df2 = df.copy()
        df2.columns = [prefix + col for col in df2.columns]
        return df2

    def load_data(self):
        df_trans, df_item, df_error = self.load_local()

        df_trans = self._add_prefix_to_columns(df_trans, 'transaction_')
        df_item = self._add_prefix_to_columns(df_item, 'item_')
        df_error = self._add_prefix_to_columns(df_error, 'error_')

        sub_df = pd.merge(df_trans, df_item, left_on='transaction_Id', right_on='item_ClaimTransactionID', how='left')
        merged_df = pd.merge(sub_df, df_error, left_on='transaction_Id', right_on='error_ClaimTransactionId', how='left')

        df_request = merged_df[merged_df['transaction_TransactionType']  == 'Request']
        df_response = merged_df[merged_df['transaction_TransactionType'] != 'Request']

        return df_request, df_response

    def merge_item_trans(self,df_item=None, df_trans=None):
        if df_item == None or df_trans == None:
            df_trans, df_item, _ = self.load_local()
        df_trans = self._get_transaction_requests(df_trans)

        df_trans = self._add_prefix_to_columns(df_trans, 'transaction_')
        df_item = self._add_prefix_to_columns(df_item, 'item_')
        merged_df = pd.merge(df_trans, df_item, left_on='transaction_RequestId', right_on='item_ClaimRequestID', how='right')
        return merged_df


class MergedDataPreprocessing:
    def __init__(self, df):
        self.df = df
        self.lstm_embedding = LSTMEmbedding()

    def _truncate_column_values(self, column):
        df = self.df.copy()
        df[column] = df[column].astype(str).str[:7]

        return df

    def _eliminate_null_claims(self,df_sorted):
        return df_sorted[df_sorted['item_ResponseState'].notnull()] ## assert 'Accepted' and 'Rejected' cases

    def train_test_split(self, id_column='transaction_RequestId', test_size=0.2, random_state=None):
        df = self.df
        df = self._eliminate_null_claims(df) ## assert 'Accepted' and 'Rejected' cases

        unique_ids = df[id_column].unique()
        train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

        train_data = df[df[id_column].isin(train_ids)]
        test_data = df[df[id_column].isin(test_ids)]

        return train_data, test_data

    def _extract_age(self,age_string: str):
        y_index = age_string.find('Y')
        years_str = age_string[:y_index]
        years = int(years_str)

        return years

    def _label_encode_column(self, column_name, min_count, replace_value='Other'):
        df = self.df.copy()
        counts = df[column_name].value_counts()
        values_to_replace = counts[counts < min_count].index
        df[column_name] = df[column_name].replace(values_to_replace, replace_value)

        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])

        return df[column_name].values

    def _process_gender(self,gender_string: str):
        if gender_string == 'Female':
            return 0
        else:
            return 1

    def _get_parent_family(self, icd10_code):
        icd10_code = str(icd10_code)
        if icd10_code in ['NaN','Nan','None','NULL','nan']:
            return 'OTHER'
        if icd10_code == 'XX':
            return 'OTHER'
        if icd10_code[0].upper() == "A" or icd10_code[0].upper() == "B":
            return 'A00–B99'
        elif icd10_code[0].upper() == "C" or (icd10_code[0].upper() == "D" and int(icd10_code[1].upper()) < 5):
            return 'C00–D48'
        elif icd10_code[0].upper() == "D":
            return 'D50–D89'
        elif icd10_code[0].upper() == "E":
            return 'E00–E90'
        elif icd10_code[0].upper() == "F":
            return 'F00–F99'
        elif icd10_code[0].upper() == "G":
            return 'G00–G99'
        elif icd10_code[0].upper() == "H" and int(icd10_code[1].upper()) < 6:
            return 'H00–H59'
        elif icd10_code[0].upper() == "H":
            return 'H60–H95'
        elif icd10_code[0].upper() == "I":
            return 'I00–I99'
        elif icd10_code[0].upper() == "J":
            return 'J00–J99'
        elif icd10_code[0].upper() == "K":
            return 'K00–K93'
        elif icd10_code[0].upper() == "L":
            return 'L00–L99'
        elif icd10_code[0].upper() == "M":
            return 'M00–M99'
        elif icd10_code[0].upper() == "N":
            return 'N00–N99'
        elif icd10_code[0].upper() == "O":
            return 'O00–O99'
        elif icd10_code[0].upper() == "P":
            return 'P00–P96'
        elif icd10_code[0].upper() == "Q":
            return 'Q00–Q99'
        elif icd10_code[0].upper() == "R":
            return 'R00–R99'
        elif icd10_code[0].upper() == "S" or icd10_code[0].upper() == "T":
            return 'S00–T98'
        elif icd10_code[0].upper() == "V" or icd10_code[0].upper() == "Y":
            return 'V01–Y98'
        elif icd10_code[0].upper() == "Z":
            return 'Z00–Z99'
        else:
            return 'U00–U99'

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

    def _map_age_range(self,age_range):
        mapping = {
        'Kids (0-2)': 1,
        'Age 3-10': 2,
        'Age 11-17': 3,
        'Age 18-24': 4,
        'Age 25-34': 5,
        'Age 35-44': 6,
        'Age 45-54': 7,
        'Age 55-64': 8,
        'Age 65+': 9
        }
        return mapping[age_range]

    def age_gender_item_ids_prep(self,item_encoding=True):

        self.df.transaction_PatientAge = self.df.transaction_PatientAge.apply(self._extract_age)
        self.df['PatientAgeRange'] = self.df.transaction_PatientAge.apply(self._categorize_age)
        self.df['PatientAgeRange'] = self.df.PatientAgeRange.apply(self._map_age_range)

        self.df.transaction_PatientEnGender = self.df.transaction_PatientEnGender.apply(self._process_gender)
        self.df.transaction_DiagnosisIds = self.df.transaction_DiagnosisIds.apply(self._get_parent_family)
        self.df.transaction_DiagnosisIds = self._label_encode_column(column_name='transaction_DiagnosisIds', min_count=100)
        if item_encoding:
            self.df.item_NameEn = self._label_encode_column(column_name='item_NameEn', min_count=15)

        self.df['item_Diagnosis'] = self.df.groupby('transaction_DiagnosisIds')['item_Price'].transform('mean')

        self.df['item_Price'] = self.df.apply(
            lambda row: row['item_NameEn'] if pd.isnull(row['item_Price']) and not pd.isnull(row['item_NameEn']) else row['item_Price'],axis=1)

        return self.df

    def column_embedding(self, df1, textual_col='item_NameEn'):
        arr2 = self.lstm_embedding.embedding_vector(df1[:], reload_model=True)
        new_cols_names = [textual_col + str(i + 1) for i in range(arr2.shape[1])]
        df2 = pd.DataFrame(arr2)
        df2.columns = new_cols_names
        for col in df2.columns:
            df1.loc[:, col] = df2[col].values

        return df1