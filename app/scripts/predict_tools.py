from app.core.config import get_api_settings
import numpy as np
import pandas as pd
import datetime
from app.scripts.general_tools import change2count, find_best_draw
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
    
    df = pd.read_csv(DATASET_FILE, sep=";")
    df = df.sort_values(by="Date")
    df_count = await change2count(df)
    y_pred_prob = model.predict_proba([df_count.iloc[-1]])
    draw_pred_prob = np.array(y_pred_prob)[:,0,1]
    np_data = np.array([data.n1,data.n2,data.n3,data.n4,data.n5,data.e1,data.e2],dtype=int)
    draw_pred_prob[:50] = np.divide(draw_pred_prob[:50],np.sum(draw_pred_prob[:50]))
    draw_pred_prob[50:] = np.divide(draw_pred_prob[50:],np.sum(draw_pred_prob[50:]))
    n_index = np_data[:5] - 1
    e_index = np_data[5:] + 50 - 1
    win_proba = np.prod(draw_pred_prob[[*n_index, *e_index]])
    proba = DrawProba(win= win_proba, lose= 1 - win_proba)
    return proba

async def generate_best_draw(model: MODEL_TYPE)-> DataLine:
    """ Find the draw with the higher probability to be the right one

    Args:
        model (MODEL_TYPE): model from sklearn library already fitted

    Returns:
        DataLine: keys : n1, n2, n3, n4, n5, e1, e2, type of values int, for n between 1 and 50, for e between 1 and 12
    """
    df = pd.read_csv(DATASET_FILE, sep=";")
    df = df.sort_values(by="Date")
    df_count = await change2count(df)
    y_pred_prob = model.predict_proba([df_count.iloc[-1]])
    draw_pred_prob = np.array(y_pred_prob)[:,0,1]
    np_best_draw = await find_best_draw(draw_pred_prob)
    best_draw = DataLine(date = str(datetime.date.today()),
                         n1 = np_best_draw[0],
                         n2 = np_best_draw[1],
                         n3 = np_best_draw[2],
                         n4 = np_best_draw[3],
                         n5 = np_best_draw[4],
                         e1 = np_best_draw[5],
                         e2 = np_best_draw[6])
    return best_draw