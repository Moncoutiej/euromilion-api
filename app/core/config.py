from functools import lru_cache
from typing import List
from sklearn.ensemble import RandomForestClassifier

import os

from pydantic import BaseSettings

class APISettings(BaseSettings):

    title = "euromillion-api"
    contacts = "moncoutiej@cy-tech.fr, urgellbapt@cy-tech.fr"

    docs_url = "/docs"
    redoc_url = "/redoc"
    
    
    api_predict_route: str = "/api/predict"
    api_model_route: str = "/api/model"


    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."

    backend_cors_origins_str: str = ""  # Should be a comma-separated list of origins
    
    model_type = RandomForestClassifier
    
    ########################     Path for data, model, ...     ########################
    
    data_csv: str = ROOT_DIR + "/data/EuroMillions_numbers.csv"
    model_file: str = ROOT_DIR + "/app/models/model.pkl"
    metrics_json: str = ROOT_DIR + "/app/models/metrics.json"

    @property
    def backend_cors_origins(self) -> List[str]:
        return [x.strip() for x in self.backend_cors_origins_str.split(",") if x]


@lru_cache()
def get_api_settings() -> APISettings:
    """Init and return the API settings

    Returns:
        APISettings: The settings
    """
    return APISettings()  # reads variables from environment