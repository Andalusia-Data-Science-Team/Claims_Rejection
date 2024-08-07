from datetime import datetime, timedelta
import urllib
import sqlalchemy
import pandas as pd
import json

with open('..\Claims_Rejection\src\data_backup\passcode.json', 'r') as file:
    data_dict = json.load(file)

db_names = data_dict['DB_NAMES']

with open('..\Claims_Rejection\src\data_backup\stored_info.json', 'r') as file:
    table_info_dict = json.load(file)

def get_connection_from_source(source):
    passcodes = db_names[source]
    server, db, uid, pwd, driver = passcodes['Server'],passcodes['Database'],passcodes['UID'],passcodes['PWD'],passcodes['driver']

    conn_str = f'''
    DRIVER={driver};Server={server};Database={db};UID={uid};PWD={pwd};charset="utf-8";
    '''
    return conn_str

def load_query(TABLE_NAME='Claim_Visit', source='BI'):
    """
    :param TABLE_NAME: Claim_Visit,Claim_Service, purchaser_Contract, Insurance_AHJ, Diagnosis
    :param source: BI -> BI-03
    :return: df
    """
    conn_str = get_connection_from_source(source)

    if f'{TABLE_NAME.lower()}_columns' in table_info_dict.keys():
        # Example list of columns
        table_columns = table_info_dict[f'{TABLE_NAME.lower()}_columns']

        # Convert the list to a comma-separated string
        columns_string = ", ".join(table_columns)
    else:
        # handle the case of unfiltered tables
        columns_string = '*'


    query = f'''SELECT  {columns_string}  FROM DWH_Claims.dbo.{TABLE_NAME}'''

    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)

def load_query_by_date(TABLE_NAME, LAST_DATE,source='BI'):
    conn_str = get_connection_from_source(source)

    if f'{TABLE_NAME.lower()}_columns' in table_info_dict.keys():
        table_columns = table_info_dict[f'{TABLE_NAME.lower()}_columns']
        columns_string = ", ".join(table_columns)
    else:
        columns_string = '*'

    sixty_days_before = datetime.today() - timedelta(days=60)

    if LAST_DATE > sixty_days_before:
        LAST_DATE = sixty_days_before - timedelta(days=1)
    query = f'''SELECT  {columns_string}  FROM DWH_Claims.dbo.{TABLE_NAME}
        WHERE CREATION_DATE > {LAST_DATE} AND CREATION_DATE < {sixty_days_before}'''

    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)


def update_data(LAST_DATE):
    df_service = load_query_by_date(TABLE_NAME='Claim_Service', LAST_DATE=LAST_DATE)
    df_visit    = load_query_by_date(TABLE_NAME='Claim_Visit', LAST_DATE=LAST_DATE)
    df_diagnose = load_query_by_date(TABLE_NAME='Diagnosis', LAST_DATE=LAST_DATE)


    return df_visit, df_service,df_diagnose

def load_claims_bisample(source='BI'):
    
    db_name = db_names[source]

    query = '''
    SELECT *
    FROM [DWH_Claims].[DBO].[Claims_Sample]'''

    df = load_query(query,db_name)
    return df

def load_claims(source='SNB',TABLE_NAME='ClaimTransaction',SAMPLE_SIZE=100):
    db_name = db_names[source]
    query = f'''SELECT TOP({SAMPLE_SIZE}) *  FROM Nphies.{TABLE_NAME}  ORDER BY CreatedDate DESC'''
    df = load_query(query, db_name)
    return df


# df_requests, df_response = load_merged(source='SNB')
#df = load_claims(source='SNB',TABLE_NAME='ClaimTransaction',SAMPLE_SIZE=50)