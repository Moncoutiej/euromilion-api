from fastapi.routing import APIRouter
from fastapi import HTTPException

from app.core.config import get_api_settings
from app.classes.models import DataLine, DrawProba
from app.scripts.predict_tools import model_prediction_on_data, generate_best_draw
import pickle

settings = get_api_settings()
API_PREDICT_ROUTE = settings.api_predict_route
MODEL_FILE = settings.model_file

PredictRouter = APIRouter()

@PredictRouter.post(API_PREDICT_ROUTE, response_model=DrawProba)
async def predict_on_data(data: DataLine) -> DrawProba:
    """Launch a prediction on the giving data 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (DrawProba): The Draw win and lose probabilities
    """
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        preds = await model_prediction_on_data(model, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model prediction : {e}")
    return preds


@PredictRouter.get(API_PREDICT_ROUTE, response_model=DataLine)
async def get_best_draw() -> DataLine:
    """Generate the Draw which maximize the chance to win 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (DataLine): The Draw generated
    """
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        draw = await generate_best_draw(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the draw generation : {e}")
    return draw
