import pandas as pd
import numpy as np

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