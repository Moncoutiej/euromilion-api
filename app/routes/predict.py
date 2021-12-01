from fastapi.routing import APIRouter
from app.core.config import get_api_settings

settings = get_api_settings()
API_PREDICT_ROUTE = settings.api_predict_route


PredictRouter = APIRouter()

@PredictRouter.get(API_PREDICT_ROUTE)
async def get_best():
    """Give technical informations about the Machine Learning Model used in this API 

    Returns:
        MLModel: Machine Learning Model Object
    """

    model = {"name": "Random Forest", "metric": {"name": "acc"}, "params" : {"depth" : 3}}
    
    return model
