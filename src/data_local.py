import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    def _truncate_column_values(self, column):
        df = self.df.copy()
        df[column] = df[column].astype(str).str[:7]

        return df

    def train_test_split(self, test_size=0.2, startify_column='item_CreatedDate'):
        self.df = self._truncate_column_values(column=startify_column)
        df_sorted = self.df.sort_values(by=startify_column)
        df_sorted = df_sorted[df_sorted['item_ResponseState'].notnull()] ## assert 'Accepted' and 'Rejected' cases
        train_df, test_df = train_test_split(df_sorted, test_size=test_size,
                                             stratify=df_sorted[startify_column])

        return train_df, test_df
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

    def age_gender_item_ids_prep(self):

        self.df.transaction_PatientAge = self.df.transaction_PatientAge.apply(self._extract_age)
        self.df.transaction_PatientEnGender = self.df.transaction_PatientEnGender.apply(self._process_gender)
        self.df.item_NameEn = self._label_encode_column(column_name='item_NameEn', min_count=15)
        self.df.transaction_DiagnosisIds = self.df.transaction_DiagnosisIds.apply(self._get_parent_family)
        self.df.transaction_DiagnosisIds = self._label_encode_column(column_name='transaction_DiagnosisIds', min_count=100)

        # TODO: must group-by before filling with average
        column_average = self.df.item_Price.mean()
        self.df.item_Price.fillna(column_average, inplace=True)

        return self.df