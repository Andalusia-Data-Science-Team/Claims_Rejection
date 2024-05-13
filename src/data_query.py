import urllib
import sqlalchemy
import pandas as pd
import json

with open('src\data_backup\passcode.json', 'r') as file:
    data_dict = json.load(file)
db_names = data_dict['DB_NAMES']

def get_connection_from_source(source):
    passcodes = data_dict[source]
    server, db, uid, pwd, driver = passcodes['Server'],passcodes['Database'],passcodes['UID'],passcodes['PWD'],passcodes['driver']

    conn_str = f'''
    DRIVER={driver};Server={server};Database={db};UID={uid};PWD={pwd};charset="utf-8";
    '''
    return conn_str

def load_query(query,source='BI'):
    conn_str = get_connection_from_source(source)

    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)


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