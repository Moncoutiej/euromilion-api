from app.core.config import get_api_settings
import numpy as np
import pandas as pd
from app.scripts.general_tools import change2count
from app.classes.models import DataLine, DrawProba

settings = get_api_settings()

DATASET_FILE = settings.data_csv
MODEL_TYPE = settings.model_type

async def model_prediction_on_data(model: MODEL_TYPE, data: DataLine)-> DrawProba:
    """ Calcul probability to win with numbers passed in data

    Args:
        model (MODEL_TYPE): model from sklearn library already fitted
        data (DataLine): keys : n1, n2, n3, n4, n5, e1, e2, type of values int, for n between 1 and 50, for e between 1 and 12

    Returns:
        dict: keys win and lose, win contains proba that the data draw is true
    """
    proba: DrawProba
    df = pd.read_csv(DATASET_FILE, sep=";")
    df = df.sort_values(by="Date")
    df_count = await change2count(df)
    y_pred_prob = model.predict_proba(df_count.iloc[-1])
    draw_pred_prob = np.array(y_pred_prob)[:,1]
    np_data = np.array([data["n1"],data["n2"],data["n3"],data["n4"],data["n5"],data["e1"],data["e2"]],dtype=int)
    n_index = np_data[:5] - 1
    e_index = np_data[5:] + 50 - 1
    proba['win'] = np.prod(draw_pred_prob[[*n_index, *e_index]])
    proba['lose'] = 1 - proba['lose']
    return proba

async def generate_best_draw(model: MODEL_TYPE)-> DataLine:
    """ Find the draw with the higher probability to be the right one

    Args:
        model (MODEL_TYPE): model from sklearn library already fitted

    Returns:
        DataLine: keys : n1, n2, n3, n4, n5, e1, e2, type of values int, for n between 1 and 50, for e between 1 and 12
    """
    best_draw: DataLine
    df = pd.read_csv(DATASET_FILE, sep=";")
    df = df.sort_values(by="Date")
    df_count = await change2count(df)
    y_pred_prob = model.predict_proba(df_count.iloc[-1])
    draw_pred_prob = np.array(y_pred_prob)[:,1]
    n_index =  np.argpartition(draw_pred_prob[:50], -5)[-5:]
    e_index = np.argpartition(draw_pred_prob[-12:], -2)[-2:]
    n_index += 1
    e_index += 1 
    best_draw = {'n1': n_index[0],
                 'n2': n_index[1],
                 'n3': n_index[2],
                 'n4': n_index[3],
                 'n5': n_index[4],
                 'e1': e_index[0],
                 'e2': e_index[1]}
    return best_draw