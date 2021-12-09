from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os, csv, json
import pandas as pd
import numpy as np
from app.scripts.general_tools import change2bool, change2count
from app.classes.models import DataLine

settings = get_api_settings()

DATASET_FILE = settings.data_csv
METRICS_FILE = settings.metrics_json
MODEL_TYPE = settings.model_type

async def get_metrics(model: MODEL_TYPE):
    return [{"metric_name": "accuracy", "value": 0}]

async def set_metrics(model, X_test, y_test)->None:
    metrics = {"metric_name": "accuracy", "value": 0}
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)
    return None

async def add_data_to_dataset(data: DataLine)->None:
    """[summary]

    Args:
        data (dict): keys : n1, n2, n3, n4, n5, e1, e2, type of values int, for n between 1 and 50, for e between 1 and 12
                     key : date, value date
                     key : winner, gain value int or none

    Returns:
        [type]: [description]
    """
    with open(DATASET_FILE, 'a') as f:
        writer = csv.writer(f, delimiter=';')
        f.writerow([data["date"],data["n1"],data["n2"],data["n3"],data["n4"],data["n5"],data["e1"],data["e2"],data["winner"],data["gain"]])
    return None

async def launch_model_fitting(model: MODEL_TYPE)->MODEL_TYPE:
    """ preprocess data and fit model variable

    Args:
        model (MODEL_TYPE): model from sklearn library

    Returns:
        MODEL_TYPE: model from sklearn library fitted
    """
    df = pd.read_csv(DATASET_FILE, sep=";")
    df = df.sort_values(by="Date")
    df_bool = await change2bool(df)
    df_count = await change2count(df)
    df_count = pd.DataFrame(MinMaxScaler().fit_transform(df_count), columns=df_count.columns)
    X = df_count.iloc[:-1]
    y = df_bool.iloc[1:]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
    model.fit(X_train,y_train)
    await set_metrics(model, X_test, y_test)
    return model