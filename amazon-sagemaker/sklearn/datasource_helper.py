import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
import boto3
from io import StringIO
from datetime import datetime
import boto3
import numpy as np
import os
from sklearn.externals import joblib
import tarfile


def read_from_vertica(username, password, sql_query, database=None, params=None, parse_dates=None):
    vertica_url = f"vertica+vertica_python://{username}:{password}@vertica.chotel.com:5433" + ("" if not database else "/" + database)
    engine = sa.create_engine(vertica_url)
    connection = engine.connect()
    df = pd.read_sql(sql=sql_query, con=connection, params=params, parse_dates=parse_dates)
    return df


def read_from_redshift(username, password, sql_query, database="dap", params=None, parse_dates=None):
    redshift_url = f"redshift+psycopg2://{username}:{password}@dap-redshift-proxy.prod.aws.chotel.com:{5439}" + ("" if not database else "/" + database)
    engine = sa.create_engine(redshift_url, connect_args={'sslmode': 'prefer'})
    connection = engine.connect()
    df = pd.read_sql(sql=sql_query, con=connection, params=params, parse_dates=parse_dates)
    return df


def read_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    #pd.read_csv(obj['Body'])
    return obj
    
    
def write_df_to_s3_csv(df, project_name, channel, sep=",", header=False, delete_exist_files=False):
    BUCKET_NAME = "choice-mlflow-input"
    config = TransferConfig(multipart_threshold=50000)
    s3 = boto3.client('s3')
    datetimestr = datetime.now().strftime("%m%d%Y%H%M")
    key = "{}/{}".format(project_name, channel.lower())
    file_name = "{}_{}.csv".format(channel.lower(), datetimestr)
    path = "{}/{}".format(key, file_name)
    local_file_path = "./{}".format(file_name)
            
    if delete_exist_files:
        print("Deleting existing files under {}/{}".format(BUCKET_NAME, key))
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)   
    
    # Use multipart upload (This may change in future to use file object type with upload_fileobj)
    print("Writing {} records to S3 - {}/{} Started at: {}".format(len(df), BUCKET_NAME, path, datetime.now()))
    df.to_csv(local_file_path, sep=sep, header=False)
    s3.upload_file(local_file_path, BUCKET_NAME, path)
    os.remove(local_file_path)
    print("Uploaded {} file to S3 at {}".format(file_name, datetime.now()))

    
def print_file_locations(project_name):
    s3 = boto3.client('s3')
    resp = s3.list_objects_v2(Bucket="choice-mlflow-input", Prefix=project_name)
    for obj in resp['Contents']:
        files = obj['Key']
        print(files)

        
def download_trained_model_from_s3(key):
    s3 = boto3.client('s3')
    model_file_name = key.split('/')[-1]
    s3.download_file('choice-mlflow-input', key, model_file_name)
    my_tar = tarfile.open(model_file_name)
    my_tar.extractall('.')
    return joblib.load("model.joblib")