import pandas as pd

class DataFrameRejection:
  def __init__(self, df):
    self.df = df
    self.df_label = pd.read_csv('data/HJH/messages_bart_label_map.csv')

  def map_rejection_label(self):
    df = self.df
    df['BART_LABEL'] = 'None'

    # Iterate through each row in the DataFrame and perform the label assignment
    for i in range(len(self.df)):
      note1 = df['NOTES'].iloc[i]
      if note1 in self.df_label['messages'].values:
        df['BART_LABEL'].iloc[i] = self.df_label[self.df_label['messages'] == note1]['zshot_bart_category'].values[0]

    self.df = df

  def get_label_rejection(self):
      df_label = self.df
      labels = []
      for i in range(len(df_label)):
        label_prep = df_label['BART_LABEL'].iloc[i]
        if label_prep == 'Medical Justification Denials':
          label = 1
        else:
          label = 0
        labels.append(label)
      df_label['BART_LABEL'] = labels
      return df_label