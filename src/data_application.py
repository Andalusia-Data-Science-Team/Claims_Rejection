import pandas as pd

model_columns = ['item_CreatedDate','transaction_RequestId','transaction_PatientAge', 'transaction_PatientEnGender','item_NameEn','item_Price','item_Status','item_Sequence','item_RequestQuantity','item_ResponseQuantity','transaction_DiagnosisIds','transaction_PhysicianIds','item_ResponseState']
train_columns = ['transaction_PatientAge','transaction_PatientEnGender', 'item_NameEn', 'item_Price', 'item_Sequence', 'item_RequestQuantity', 'transaction_DiagnosisIds', 'item_ResponseState']

class DataFrameProcessor:
    def __init__(self, df):
        self.df = df

    def get_types(self):
        df2 = self.df.dtypes.reset_index()
        df2.columns = ['column_name', 'data_type']
        return df2

    def get_cols_lowvar(self):
        unique_counts = self.df.nunique()
        single_value_columns = unique_counts[unique_counts <= 1].index.tolist()
        return single_value_columns

    def drop_columns_with_string(self, substring):
        columns_to_drop = [col for col in self.df.columns if substring in col]
        df2 = self.df.drop(columns=columns_to_drop)
        return df2

    def filter_high_variance_features(self):
        low_vars = list(self.get_cols_lowvar())
        high_var_cols = [col for col in self.df.columns if col not in low_vars]
        self.df = self.df[high_var_cols]
        self.df = self.drop_columns_with_string('Ar')
        return self.df


### Use Case
# processor = DataFrameProcessor(df)
# types_df = processor.get_types()
# df = processor.filter_high_variance_features()