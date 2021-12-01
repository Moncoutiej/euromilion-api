from functools import lru_cache
from typing import List

from pydantic import BaseSettings

class APISettings(BaseSettings):

    title = "euromilion-api"
    contacts = "moncoutiej@cy-tech.fr, urgellbapt@cy-tech.fr"

    docs_url = "/docs"
    redoc_url = "/redoc"
    
    
    api_predict_route: str = "/api/predict"
    api_model_route: str = "/api/model"


    backend_cors_origins_str: str = ""  # Should be a comma-separated list of origins

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