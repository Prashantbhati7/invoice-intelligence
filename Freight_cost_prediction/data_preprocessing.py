import sqlite3
from sklearn.model_selection import train_test_split 
import pandas as pd

def load_vendor_invoice_data(db_paht:str):
    conn = sqlite3.connect(db_paht)
    query = 'SELECT * from vendor_invoice'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features(df:pd.DataFrame):
    X = df[['Dollars','Quantity']]
    Y= df['Freight']
    return X,Y

def split_data(X,Y,test_size=0.2,random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state= random_state)
    return X_train, X_test, Y_train, Y_test