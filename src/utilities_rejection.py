import re

class DataFrameRejection:
  def __init__(self, df):
    self.df = df

  def map_rejection_label(self):
    df = self.df

    medical_pattern = r"MN-1-[1-2]"
    admin_pattern   = r"AD-[1-3]-[1-8]"
    cover_pattern   = r"CV-[1-3]-(10|[1-9])"
    support_pattern = r"SE-1-(10|[1-9])"
    billing_pattern = r"BE-1-(10|[1-9])"

    # Iterate through each row in the DataFrame and perform the label assignment
    labels = []
    for i in range(len(df)):
        note1 = df['NOTES'].iloc[i]
        if re.search(medical_pattern, note1):
            label_nphies = 'Medical'
        elif re.search(admin_pattern, note1):
            label_nphies = 'Administrative'
        elif re.search(cover_pattern, note1):
            label_nphies = 'Coverage'
        elif  re.search(support_pattern, note1):
            label_nphies = 'Support'
        elif  re.search(billing_pattern, note1):
            label_nphies = 'Billing'
        else:
            label_nphies = 'NoLabel'
        labels.append(label_nphies)

    df['NPHIES_LABEL'] = labels
    self.df = df
    return df

  def get_label_rejection(self):
      df_label = self.df
      labels = []
      for i in range(len(df_label)):
        label_prep = df_label['NPHIES_LABEL'].iloc[i]
        if label_prep == 'Medical':
          label = 1
        else:
          label = 0
        labels.append(label)
      df_label['NPHIES_LABEL'] = labels
      return df_label