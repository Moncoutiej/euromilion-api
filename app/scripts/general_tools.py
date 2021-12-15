from app.classes.models import DataLine
import pandas as pd
import numpy as np
import datetime

async def change2bool(df:pd.DataFrame)->pd.DataFrame:
    """ Construct vectors of a draw for each date

    Args:
        df (pd.DataFrame): DataFrame from EuroMillions_numbers.csv ordered by ascending Date

    Returns:
        pd.DataFrame: Dataframe of model features after the preprocessing phase
    """
    df_N_E_bool: pd.DataFrame
    
    np_df = df.to_numpy()
    arr_N_E_bool = np.zeros((df.shape[0],50+12))
    for i in range(np_df.shape[0]):
        N_E_bool = np.zeros(50+12, dtype=np.int)
        index = np.add(np_df[i,1:6],-1)
        index = np.array(index,dtype=int)
        N_E_bool[index] = 1
        index = np.add(np_df[i,6:8],-1+50)
        index = np.array(index,dtype=int)
        N_E_bool[index] = 1
        arr_N_E_bool[i] = N_E_bool
        
    columns = list(range(1,50+12+1))
    index = pd.MultiIndex.from_arrays([df["Date"],df.index],names=["Date","Index"])
    df_N_E_bool = pd.DataFrame(arr_N_E_bool, columns = columns, index = index)
    return df_N_E_bool

async def change2count(df:pd.DataFrame)->pd.DataFrame:
    """ Give the occurrence frequency of numbers date by date and subtract occurence with the min frequency for each row

    Args:
        df (pd.DataFrame): DataFrame from EuroMillions_numbers.csv ordered by ascending Date

    Returns:
        pd.DataFrame: Dataframe of model features after the preprocessing phase
    """
    df_N_E_compteur: pd.DataFrame
    
    np_df = df.to_numpy()
    N_E_compteur = np.zeros(50+12, dtype=np.int)
    arr_N_E_compteur = np.zeros((df.shape[0],50+12))
    for i in range(np_df.shape[0]):
        index = np.add(np_df[i][1:6],-1)
        index = np.array(index,dtype=int)
        N_E_compteur[index] += 1
        index = np.add(np_df[i][6:8],-1+50)
        index = np.array(index,dtype=int)
        N_E_compteur[index] += 1
        N_min = np.amin(N_E_compteur[:50])
        E_min = np.amin(N_E_compteur[50:])
        N_compteur = np.subtract(N_E_compteur[:50],N_min,dtype=int)
        E_compteur = np.subtract(N_E_compteur[50:],E_min,dtype=int)
        arr_N_E_compteur[i,:50] = N_compteur
        arr_N_E_compteur[i,50:] = E_compteur
        
    columns = list(range(1,50+12+1))
    index = pd.MultiIndex.from_arrays([df["Date"],df.index],names=["Date","Index"])
    df_N_E_compteur = pd.DataFrame(arr_N_E_compteur, columns = columns, index = index)
    return df_N_E_compteur


async def verify_user_data(data: DataLine) -> str:
    """Verify the user input and returns an apropriate message if the given data is incorect

    Args:
        data (DataLine): The Data given in input

    Returns:
        str: The apropriate message given to user if the data is incorrect. returns "" if the input is correct 
    """
    message: str = ''
    
    draw_numbers = [data.n1, data.n2, data.n3, data.n4, data.n5]
    draw_stars = [data.e1, data.e2]
    
    # If the draw numbers are non unic in the input
    if len(draw_numbers) > len(set(draw_numbers)) or len(draw_stars) > len(set(draw_stars)) : 
        message += 'The values given for this draw must be unical. '
        
    # If the draw numbers are non unic in the input
    try:
        datetime.datetime.strptime(data.date, '%Y-%m-%d')
    except Exception:
        message += 'Incorrect date, should be a valid YYYY-MM-DD'
    
    return message