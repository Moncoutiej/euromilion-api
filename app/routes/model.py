from fastapi.routing import APIRouter
from app.core.config import get_api_settings
from app.classes.models import MLModel

settings = get_api_settings()
API_MODEL_ROUTE = settings.api_model_route

ModelRouter = APIRouter()


@ModelRouter.get(API_MODEL_ROUTE)
async def get_model() -> MLModel:
    """Give technical informations about the Machine Learning Model used in this API 

    Returns:
        MLModel: Machine Learning Model Object
    """

    model = {"name": "Random Forest", "metric": {"name": "acc"}, "params" : {"depth" : 3}}
    
    return model

