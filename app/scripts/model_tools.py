from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import csv, json
from typing import Dict, List
import pandas as pd
import numpy as np
from app.scripts.general_tools import change2bool, change2count, find_best_draw
from app.classes.models import DataLine, Metric

settings = get_api_settings()

DATASET_FILE = settings.data_csv
METRICS_FILE = settings.metrics_json
MODEL_TYPE = settings.model_type

async def get_metrics()->List[Metric]:
    """ Read json file and format data to match with List[Metric] type

    Returns:
        List[Metric]: list of metric for the current model app
    """
    metrics: List[Metric] = []
    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)
    for metric_name, value in data.items():
           metrics.append(Metric(metric_name=metric_name, value=value))
    return metrics

async def custom_predict(y_pred_prob:np.ndarray)->np.ndarray:
    """ Find the best draw for each prediction given in parameter

    Args:
        y_pred_prob (np.ndarray): result of X_test predict probability

    Returns:
        np.ndarray: array of the best draw for each predictions
    """
    y_pred: np.ndarray
    y_pred = np.zeros((y_pred_prob.shape[1],y_pred_prob.shape[0]))
    for i in range(y_pred_prob.shape[1]):
        y_prob = y_pred_prob[:,i,1]
        draw = await find_best_draw(y_prob)
        index = draw - 1
        index[-2:] += 50
        y_pred[i,index] = 1
    return y_pred

async def set_metrics(model: MODEL_TYPE, X_test: pd.DataFrame, y_test: pd.DataFrame)->None:
    """ Calcul and save in a json file differents metrics of the model

    Args:
        model (MODEL_TYPE): model from sklearn library
        X_test (pd.DataFrame): values for testing the model
        y_test (pd.DataFrame): output of the testing value

    Returns:
        [type]: Nothing
    """
    metrics: Dict[str,float] = {}
    
    y_pred_prob = model.predict_proba(X_test)
    np_y_pred_prob = np.array(y_pred_prob)
    custom_y_pred = await custom_predict(np_y_pred_prob)
    recall_value = recall_score(y_test, custom_y_pred, average="weighted")
    metrics["Recall"] = recall_value
    precision_value = precision_score(y_test, custom_y_pred, average="weighted")
    metrics["Precision"] = precision_value
    f1_value = f1_score(y_test, custom_y_pred, average="weighted")
    metrics["F1"] = f1_value 
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)
    return None

async def add_data_to_dataset(data: DataLine)->None:
    """ Add a new row at the end of the EuroMillions_numbers.csv

    Args:
        data (dict): keys : n1, n2, n3, n4, n5, e1, e2, type of values int, for n between 1 and 50, for e between 1 and 12
                     key : date, value date
                     key : winner, gain value int or none

    Returns:
        [type]: Nothing
    """
    with open(DATASET_FILE, 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([data.date,data.n1,data.n2,data.n3,data.n4,data.n5,data.e1,data.e2,data.winner,data.gain])
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