from fastapi.routing import APIRouter
from fastapi import HTTPException

from app.routes.model import train_model
from app.core.config import get_api_settings
from app.classes.models import DataLine, DrawProba
from app.scripts.predict_tools import model_prediction_on_data, generate_best_draw
from app.scripts.general_tools import verify_user_data
import pickle, os

settings = get_api_settings()
API_PREDICT_ROUTE = settings.api_predict_route
MODEL_FILE = settings.model_file

PredictRouter = APIRouter()

@PredictRouter.post(API_PREDICT_ROUTE, response_model=DrawProba)
async def predict_on_data(data: DataLine) -> DrawProba:
    """Launch a prediction on the giving data 

    Raises:
        HTTPException: 400 status code if the user input is incorrect
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (DrawProba): The Draw win and lose probabilities
    """
    message = await verify_user_data(data)
    if len(message) > 0:
        raise HTTPException(status_code=400, detail=f"Error while model prediction : {message}")
    try:
        if not os.path.exists(MODEL_FILE):
            await train_model()
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
        if not os.path.exists(MODEL_FILE):
            await train_model()
        model = pickle.load(open(MODEL_FILE, "rb"))
        draw = await generate_best_draw(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the draw generation : {e}")
    return draw
